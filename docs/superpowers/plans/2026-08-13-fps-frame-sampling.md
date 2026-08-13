# fps Frame Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `fps` frame-selection method that samples at a constant wall-clock rate, and untangle `max_frames` into a single "frame budget" meaning by deleting `frame_proportion`.

**Architecture:** `_sample_fps` joins the existing `_sample_uniform` / `_sample_optical_flow` siblings in `collab_splats/preproc/sampling.py`. The two target-list methods (`fps`, `uniform`) differ only in how they build a list of source frame indices; they share `_sample_positions`, which is today's `_sample_uniform` body (single ffmpeg `select=` pass, validation window, quality gate, sharpest-neighbour substitution) with the target block lifted out. `optical_flow` is untouched.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), numpy, OpenCV (in-memory ops only), ffmpeg/ffprobe (sole decode backend), pytest, Panel (dashboard), YAML configs.

**Spec:** `docs/superpowers/specs/2026-08-13-fps-frame-sampling-design.md`

**Run tests with:** `/opt/venv/reconstruction/bin/python -m pytest <path> -p no:randomly -v`
(`-p no:randomly` is required — `pytest-randomly` is installed and reorders tests.)

**Commit with an explicit pathspec** (`git commit -m "..." -- path1 path2`). Concurrent sessions share this repo's git index; a bare `git commit` can sweep in another session's staged files.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `collab_splats/preproc/sampling.py` | target builders, three samplers, shared decode/gate body, dispatch + validation | Modify |
| `tests/preproc/test_sampling.py` | sampler unit + behaviour tests | Modify |
| `collab_splats/wrapper/reconstructor.py` | config → sampler plumbing, `frames.zarr` provenance | Modify |
| `tests/wrapper/test_reconstructor.py` | `_extract_frames` branch tests, base.yaml default assertions | Modify |
| `tests/wrapper/test_reconstructor_preprocess.py` | end-to-end preprocess config | Modify |
| `configs/base.yaml` | default `frame_selection: fps`, new `fps`, delete `frame_proportion` | Modify |
| `configs/loop_closure.yaml` | LC override moves to `fps` + explicit floor | Modify |
| `configs/README.md` | key reference table + sampler guidance | Modify |
| `docs/source/tutorials/03_splats/configs/base.yaml` | tutorial `Reconstructor` config | Modify |
| `collab_splats/dashboard/config.py` | `RunConfig.fps` | Modify |
| `collab_splats/dashboard/app.py` | fps widget, sampler options, persistence, visibility | Modify |
| `collab_splats/dashboard/pipeline.py` | `_sample` branch, `_write_frames_zarr` provenance | Modify |

No new files. `sampling.py` gains ~55 lines and loses ~40.

---

### Task 1: Pure target builders

Two functions that turn a frame count / rate into a list of source frame indices. No video IO, so they unit-test with plain integers.

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (insert after `_progress_reporter`, which ends at line 354)
- Test: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_sampling.py`:

```python
########################################################################
# Target builders (pure — no video IO)
########################################################################


def test_uniform_targets_spans_endpoints():
    # "N frames spanning the video": first and last source frames are both included
    from collab_splats.preproc.sampling import _uniform_targets

    t = _uniform_targets(60, 5)
    assert t[0] == 0 and t[-1] == 59
    assert len(t) == 5
    assert t == sorted(t)


def test_uniform_targets_caps_at_total():
    # Asking for more frames than exist yields every frame, not duplicates
    from collab_splats.preproc.sampling import _uniform_targets

    assert _uniform_targets(4, 10) == [0, 1, 2, 3]


def test_uniform_targets_degenerate_inputs():
    from collab_splats.preproc.sampling import _uniform_targets

    assert _uniform_targets(0, 5) == []
    assert _uniform_targets(60, 0) == []


def test_fps_targets_uses_constant_stride():
    # "a frame every 1/fps seconds": 30 fps source at 10 fps → stride 3
    from collab_splats.preproc.sampling import _fps_targets

    t = _fps_targets(60, 30.0, 10.0)
    assert t == list(range(0, 60, 3))
    assert len(t) == 20


def test_fps_targets_does_not_stretch_to_last_frame():
    # Stride-anchored, NOT endpoint-anchored: spacing is the contract, so the last
    # target is wherever the stride lands — this is what distinguishes fps from uniform.
    from collab_splats.preproc.sampling import _fps_targets

    t = _fps_targets(50, 30.0, 10.0)
    assert t[-1] == 48  # not 49


def test_fps_targets_clamps_stride_to_one():
    # Requesting a rate above the source rate cannot sample sub-frame; stride floors at 1
    from collab_splats.preproc.sampling import _fps_targets

    assert _fps_targets(10, 30.0, 120.0) == list(range(10))


def test_fps_targets_degenerate_inputs():
    from collab_splats.preproc.sampling import _fps_targets

    assert _fps_targets(0, 30.0, 1.0) == []
    assert _fps_targets(60, 30.0, 0.0) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -p no:randomly -k "targets" -v`
Expected: FAIL — `ImportError: cannot import name '_uniform_targets' from 'collab_splats.preproc.sampling'`

- [ ] **Step 3: Write the implementation**

Insert into `collab_splats/preproc/sampling.py` immediately after `_progress_reporter` (after line 354) and before `def sample_frames`:

```python
def _uniform_targets(total: int, n: int) -> list[int]:
    """N evenly-spaced source indices spanning the whole video (endpoint-anchored)."""
    if total <= 0 or n <= 0:
        return []
    # Cap at the source length, then dedup: rounding can collide when n approaches total
    n = min(n, total)
    return np.unique(np.linspace(0, total - 1, n).round().astype(int)).tolist()


def _fps_targets(total: int, native_fps: float, fps: float) -> list[int]:
    """Source indices at a constant wall-clock interval (stride-anchored).

    Deliberately not stretched to hit the last frame: for fps the spacing is the
    contract and the count falls out, the mirror of _uniform_targets.
    """
    if total <= 0 or fps <= 0:
        return []
    # Stride floors at 1 — a requested rate above the source rate cannot sample sub-frame
    step = max(1, int(round((native_fps or 30.0) / fps)))
    return list(range(0, total, step))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -p no:randomly -k "targets" -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): add _uniform_targets and _fps_targets builders" -- collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
```

---

### Task 2: Extract `_sample_positions` (pure refactor)

Lift the decode + quality-gate body out of `_sample_uniform` so both target-list samplers share it. Behaviour must not change — the existing tests are the proof.

**Files:**
- Modify: `collab_splats/preproc/sampling.py:454-523` (`_sample_uniform`)

- [ ] **Step 1: Confirm the existing tests pass before touching anything**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -p no:randomly -v`
Expected: PASS — record the count; it must be identical after this task.

- [ ] **Step 2: Replace `_sample_uniform` with the split pair**

Replace the whole of `collab_splats/preproc/sampling.py:454-523` (from `def _sample_uniform(` through `    return frames, records`) with:

```python
def _sample_positions(
    video_path: str,
    targets: list[int],
    *,
    total: int,
    w: int,
    h: int,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
    desc: str,
) -> tuple[list[np.ndarray], list[dict]]:
    """Decode and quality-gate one frame per target position in a single ffmpeg pass.

    Shared body of the target-list samplers (_sample_fps, _sample_uniform). Each
    position gets a validation window of neighbouring frames; the sharpest
    gate-passing frame in the window wins, falling back to the sharpest frame when
    none passes, so the returned count stays exact.
    """
    if not targets:
        return [], []
    # Validation window: radius half the target spacing, capped, and kept under
    # spacing/2 so neighbouring windows never overlap (deterministic index map).
    spacing = targets[1] - targets[0] if len(targets) > 1 else total
    radius = min(max((spacing - 1) // 2, 0), _VALID_PROBE_MAX)
    # Per-target candidate indices (clamped to range, dedup); flatten to one
    # sorted set of frames to decode in a single pass.
    windows = [sorted({min(max(t + o, 0), total - 1) for o in range(-radius, radius + 1)}) for t in targets]
    wanted = sorted({i for win in windows for i in win})
    # One ffmpeg pass → frames keyed by source index (in-C decode, ~len(wanted)
    # frames reach Python).
    frame_by_idx = dict(zip(wanted, _iter_selected_frames(video_path, wanted, w, h)))
    report, close = _progress_reporter(len(targets), desc, on_progress)
    frames: list[np.ndarray] = []
    records: list[dict] = []
    try:
        # Per position, pick the sharpest gate-passing frame in its window; fall
        # back to the sharpest frame when none passes, so the count stays exact.
        for done, win in enumerate(windows):
            best = None  # ((usable, blur), idx, bgr) — prefer usable, then sharp
            for idx in win:
                bgr = frame_by_idx.get(idx)
                if bgr is None:
                    continue
                gray = _analysis_gray(bgr)
                blur = compute_blur_score(gray)
                usable, _ = check_frame_quality(gray, blur_threshold, blur_score=blur)
                key = (usable, blur)
                if best is None or key > best[0]:
                    best = (key, idx, bgr)
            if best is None:
                continue  # window fully dropped by ffmpeg (should not happen)
            key, idx, bgr = best
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            records.append({"frame_idx": idx, "blur_score": key[1]})
            report(done + 1)
    finally:
        close()
    return frames, records


def _sample_uniform(
    video_path: str,
    *,
    max_frames: int,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Exactly max_frames evenly-spaced frames spanning the whole video."""
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []
    targets = _uniform_targets(total, max_frames)
    return _sample_positions(
        video_path,
        targets,
        total=total,
        w=info["width"],
        h=info["height"],
        blur_threshold=blur_threshold,
        on_progress=on_progress,
        desc="Uniform sampling",
    )
```

Note what left: the `fps` parameter, the `native_fps` local, the three-way target branch, the hidden `2.0` fps literal, and the `targets = targets[:max_frames]` truncation. All are replaced by `_uniform_targets`.

- [ ] **Step 3: Update the one call site in `sample_frames` so the module imports**

In `collab_splats/preproc/sampling.py`, inside `sample_frames`, replace:

```python
    if method == "uniform":
        return _sample_uniform(
            video_path,
            fps=fps,
            max_frames=max_frames,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
```

with:

```python
    if method == "uniform":
        return _sample_uniform(
            video_path,
            max_frames=max_frames,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
```

- [ ] **Step 4: Run the sampling tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -p no:randomly -v`
Expected: the four tests that pass `fps=` alongside `method="uniform"` now FAIL with `TypeError: _sample_uniform() got an unexpected keyword argument 'fps'` (lines 213, 221, 270, 314). **Every other test must still PASS**, including `test_uniform_selects_in_one_pass_not_whole_video` and `test_uniform_best_effort_keeps_count_when_gate_rejects_all` — those prove the lifted body is unchanged. The four failures are fixed in Task 3.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/sampling.py
git commit -m "refactor(preproc): extract _sample_positions from _sample_uniform" -- collab_splats/preproc/sampling.py
```

---

### Task 3: `_sample_fps`, dispatch, and validation

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (`sample_frames` at line 357; new `_sample_fps`)
- Test: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Migrate the existing call sites and write the failing tests**

In `tests/preproc/test_sampling.py`, make these five edits.

`test_uniform_returns_frames_and_records` (line ~211) — rename and switch method:

```python
def test_fps_returns_frames_and_records(tiny_video):
    frames, records = sample_frames(tiny_video, method="fps", fps=10.0)
    # 60 frames @30fps sampled at 10fps → stride 3 → 20
    assert len(frames) == len(records) == 20
    assert frames[0].shape == (240, 320, 3)
    assert set(records[0]) == {"frame_idx", "blur_score"}
```

`test_uniform_respects_max_frames` (line ~220):

```python
def test_fps_respects_max_frames(tiny_video):
    frames, records = sample_frames(tiny_video, method="fps", fps=10.0, max_frames=5)
    assert len(frames) == len(records) == 5
```

`test_uniform_frames_are_rgb` (line ~269):

```python
def test_fps_frames_are_rgb(tiny_video):
    frames, records = sample_frames(tiny_video, method="fps", fps=10.0)
    bgr = list(_iter_frames(tiny_video))
    # RGB return means channel order is reversed vs the BGR decode
    np.testing.assert_array_equal(frames[0], bgr[records[0]["frame_idx"]][:, :, ::-1])
```

`test_sample_frames_on_progress_called` (line ~309):

```python
def test_sample_frames_on_progress_called(tiny_video):
    # Progress is reported over the selected count (fps=10 on a 2s video = 20),
    # not the whole source-frame count.
    calls = []
    frames, _ = sample_frames(
        tiny_video, method="fps", fps=10.0, on_progress=lambda done, total: calls.append((done, total))
    )
    assert calls and calls[-1] == (len(frames), len(frames)) == (20, 20)
```

`test_sample_frames_missing_file_returns_empty` (line ~352) — uniform now requires a count, and validation fires before any IO:

```python
def test_sample_frames_missing_file_returns_empty():
    frames, records = sample_frames("/nonexistent/video.mp4", method="uniform", max_frames=6)
    assert frames == [] and records == []
```

Then append the new tests:

```python
########################################################################
# fps sampler — band behaviour and cross-method validation
########################################################################


def test_fps_ceiling_decimates_rather_than_truncating(tiny_video):
    # THE regression test for `targets = targets[:max_frames]`: over-ceiling must
    # re-spread across the WHOLE video, not keep the first N and drop the tail.
    # 60 frames @30fps at 30 fps = 60 targets, capped to 10.
    frames, records = sample_frames(tiny_video, method="fps", fps=30.0, max_frames=10)
    assert len(frames) == 10
    # Truncation would put the last frame near index 9; re-spreading puts it near 59.
    assert records[-1]["frame_idx"] >= 50


def test_fps_floor_respreads_over_whole_video(tiny_video):
    # 1 fps on a 2s video = 2 targets, below the floor → re-spread at min_frames
    frames, records = sample_frames(tiny_video, method="fps", fps=1.0, min_frames=10)
    assert len(frames) == 10
    assert records[-1]["frame_idx"] >= 50


def test_fps_within_band_is_untouched(tiny_video):
    # 20 targets sits inside [5, 40] → neither guard fires, stride is preserved
    frames, records = sample_frames(tiny_video, method="fps", fps=10.0, min_frames=5, max_frames=40)
    assert len(frames) == 20
    assert records[0]["frame_idx"] == 0


def test_fps_band_warns_when_it_binds(tiny_video, caplog):
    # A bound band silently changes the effective rate — it must be logged
    import logging

    with caplog.at_level(logging.WARNING, logger="collab_splats.preproc.sampling"):
        sample_frames(tiny_video, method="fps", fps=30.0, max_frames=10)
    assert any("effective" in r.message.lower() or "effective" in r.getMessage().lower() for r in caplog.records)


def test_fps_requires_fps_value(tiny_video):
    with pytest.raises(ValueError, match="requires fps"):
        sample_frames(tiny_video, method="fps")


def test_uniform_rejects_fps_kwarg(tiny_video):
    # The exact combination this design removes: uniform's count IS max_frames
    with pytest.raises(ValueError, match="takes no fps"):
        sample_frames(tiny_video, method="uniform", max_frames=6, fps=10.0)


def test_uniform_requires_max_frames(tiny_video):
    with pytest.raises(ValueError, match="requires max_frames"):
        sample_frames(tiny_video, method="uniform")


def test_optical_flow_rejects_fps_kwarg(tiny_video):
    with pytest.raises(ValueError, match="takes no fps"):
        sample_frames(tiny_video, method="optical_flow", fps=10.0)


def test_min_frames_rejected_by_non_fps_methods(tiny_video):
    # min_frames is an fps-only floor: uniform is pinned at its count, and
    # optical_flow cannot be forced to yield frames its selector rejected.
    with pytest.raises(ValueError, match="min_frames"):
        sample_frames(tiny_video, method="uniform", max_frames=6, min_frames=3)
    with pytest.raises(ValueError, match="min_frames"):
        sample_frames(tiny_video, method="optical_flow", min_frames=3)


def test_unknown_method_raises(tiny_video):
    with pytest.raises(ValueError, match="Unknown method"):
        sample_frames(tiny_video, method="nonsense")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -p no:randomly -v`
Expected: FAIL — the `method="fps"` tests raise `ValueError: Unknown method: 'fps' ...`, and the validation tests fail because no `ValueError` is raised.

- [ ] **Step 3: Add `_sample_fps`**

Insert into `collab_splats/preproc/sampling.py` immediately after `_sample_uniform` (before `_iter_scored_frames`):

```python
def _sample_fps(
    video_path: str,
    *,
    fps: float,
    min_frames: int | None,
    max_frames: int | None,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """One frame every 1/fps seconds; re-spread if the count falls outside the band."""
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []
    native_fps = info["fps"] or 30.0
    targets = _fps_targets(total, native_fps, fps)
    # The requested rate yields a count that floats with video length, so clamp it into
    # [min_frames, max_frames] by re-spreading over the WHOLE video — never by truncating,
    # which would drop the tail of the scene and hand the reconstructor half a video.
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))
    if bounded != requested:
        targets = _uniform_targets(total, bounded)
        effective = native_fps * len(targets) / total
        logger.warning(
            "fps=%.3f wanted %d frames, outside [min_frames=%s, max_frames=%s]; re-spread to "
            "%d frames over the whole video (effective %.3f fps)",
            fps,
            requested,
            min_frames,
            max_frames,
            len(targets),
            effective,
        )
    return _sample_positions(
        video_path,
        targets,
        total=total,
        w=info["width"],
        h=info["height"],
        blur_threshold=blur_threshold,
        on_progress=on_progress,
        desc="fps sampling",
    )
```

- [ ] **Step 4: Rewrite `sample_frames` dispatch and validation**

Replace the whole of `collab_splats/preproc/sampling.py:357-405` (from `def sample_frames(` through the closing `raise ValueError(f"Unknown method: ...")`) with:

```python
def sample_frames(
    video_path: str,
    *,
    method: str = "fps",
    fps: float | None = None,
    min_frames: int | None = None,
    max_frames: int | None = None,
    min_disparity: float = 50.0,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Select frames from a video for reconstruction.

    Each method has exactly one density knob; `max_frames` is the frame budget —
    the target count for "uniform", a ceiling for the other two.

    Methods:
        "fps": one frame every 1/`fps` seconds (stride-anchored), so the baseline
            between consecutive frames is fixed regardless of video length and the
            count floats. `min_frames`/`max_frames` bound that count: outside the
            band the targets are re-spread evenly over the WHOLE video (never
            truncated) and the effective fps is logged.
        "uniform": exactly `max_frames` evenly-spaced frames spanning the video
            (endpoint-anchored) — the count is the contract and spacing falls out.
        "optical_flow": motion (LK disparity + rotation) + coverage (histogram
            diversity) scoring; frames scoring >= 0.5 are selected. Uses
            `min_disparity`; `max_frames` caps the result.

    Both target-list methods decode in a single ffmpeg `select` pass and validate
    each position against the quality gate, substituting the sharpest usable
    neighbour so the count stays exact. `blur_threshold` tunes the gate for all
    three methods (0.0 disables the blur check).

    Returns (frames, records): RGB arrays and one dict per selected frame with at
    least frame_idx (SOURCE video index) and blur_score; optical_flow adds
    disparity, rotation, histogram_similarity, score, selected.

    Raises:
        ValueError: on an unknown method, or a knob that belongs to another method.
    """
    # Reject the wrong knob for the method rather than silently ignoring it — a
    # method whose behaviour depends on which kwargs happen to be set is the defect
    # this dispatch exists to remove. Validation runs before any IO.
    if method != "fps" and fps is not None:
        raise ValueError(f"method={method!r} takes no fps= — use method='fps' to sample at a rate")
    if method != "fps" and min_frames is not None:
        raise ValueError(
            f"method={method!r} takes no min_frames= — the floor only applies to method='fps', "
            "whose count floats with video length"
        )

    if method == "fps":
        if fps is None:
            raise ValueError("method='fps' requires fps= (samples per second)")
        return _sample_fps(
            video_path,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
    if method == "uniform":
        if max_frames is None:
            raise ValueError("method='uniform' requires max_frames= (the frames to spread over the video)")
        return _sample_uniform(
            video_path,
            max_frames=max_frames,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
    if method == "optical_flow":
        return _sample_optical_flow(
            video_path,
            max_frames=max_frames,
            min_disparity=min_disparity,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
    raise ValueError(f"Unknown method: {method!r} (expected 'fps', 'uniform' or 'optical_flow')")
```

- [ ] **Step 5: Run the full sampling suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -p no:randomly -v`
Expected: PASS — all tests green, including the pre-existing `test_uniform_derives_window_from_max_frames`, `test_uniform_best_effort_keeps_count_when_gate_rejects_all`, and `test_uniform_selects_in_one_pass_not_whole_video`.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): add fps sampler with a [min_frames, max_frames] band" -- collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
```

---

### Task 4: Reconstructor plumbing

Three-way branch, `fps` threaded through, `frame_proportion` deleted, `fps` recorded in `frames.zarr` provenance.

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:65-140` (`_extract_frames`), `:498-506` (config read)
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing tests**

In `tests/wrapper/test_reconstructor.py`, replace `test_extract_frames_uniform_is_default_branch` (line ~139) with:

```python
def test_extract_frames_dispatches_per_frame_selection(tmp_path, monkeypatch):
    """Each frame_selection value reaches its own sampler with only its own knobs."""
    from collab_splats.wrapper import reconstructor as R

    calls = {}

    def fake_sample_frames(path, **kwargs):
        calls.clear()
        calls.update(kwargs)
        return [np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

    def fake_video_info(path):
        return {"total_frames": 100}

    monkeypatch.setattr(R, "sample_frames", fake_sample_frames)
    monkeypatch.setattr(R, "get_video_info", fake_video_info)

    out = tmp_path / "out"
    video = tmp_path / "v.mp4"
    video.touch()

    # fps: rate + both band bounds
    R._extract_frames(video, out, "fps", 2.0, 5, 50)
    assert calls == {"method": "fps", "fps": 2.0, "min_frames": 5, "max_frames": 50}

    # uniform: max_frames is the count; no fps, no floor
    R._extract_frames(video, out, "uniform", None, 5, 50)
    assert calls == {"method": "uniform", "max_frames": 50}

    # optical_flow: max_frames caps the selector
    R._extract_frames(video, out, "optical_flow", None, 5, 50)
    assert calls == {"method": "optical_flow", "max_frames": 50}


def test_extract_frames_rejects_unknown_selection(tmp_path, monkeypatch):
    from collab_splats.wrapper import reconstructor as R

    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    video = tmp_path / "v.mp4"
    video.touch()
    with pytest.raises(ValueError, match="frame_selection"):
        R._extract_frames(video, tmp_path / "out", "balanced", None, 5, 50)


def test_extract_frames_records_fps_in_provenance(tmp_path, monkeypatch):
    """frames.zarr must record the rate a scene was sampled at, not just the cap."""
    from collab_splats.wrapper import reconstructor as R

    monkeypatch.setattr(
        R,
        "sample_frames",
        lambda path, **kw: ([np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]),
    )
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})

    captured = {}
    real_create = R.FrameStore.create
    monkeypatch.setattr(
        R.FrameStore,
        "create",
        staticmethod(lambda path, frames, records, provenance: captured.update(provenance) or real_create(
            path, frames, records, provenance=provenance
        )),
    )

    video = tmp_path / "v.mp4"
    video.touch()
    R._extract_frames(video, tmp_path / "out" / "frames.zarr", "fps", 2.0, None, 50)
    assert captured["fps"] == 2.0
    assert captured["method"] == "fps"
```

Then update the two config fixtures and the base.yaml assertion in the same file:

- Line ~34, in `test_config_load_base_defaults`: change
  `"preprocessing": {"frame_selection": "fps", "frame_proportion": 0.1, "min_frames": 300},`
  to
  `"preprocessing": {"frame_selection": "fps", "fps": 1.0, "min_frames": 300},`
- Line ~104, in `_make_config`: change
  `"preprocessing": {"frame_selection": "uniform", "frame_proportion": 0.1, "min_frames": 10},`
  to
  `"preprocessing": {"frame_selection": "uniform", "fps": 1.0, "min_frames": 10},`
- Line ~180-181, in `test_init_fills_defaults_from_base_yaml`: change

```python
    # min_frames comes from base.yaml (150), NOT a stale code default (300)
    assert rec.config["preprocessing"]["min_frames"] == 150
```

to

```python
    # min_frames is null in base.yaml so fps is honoured literally, NOT a stale code default
    assert rec.config["preprocessing"]["min_frames"] is None
    # fps comes from base.yaml (1.0), and fps is the default selection method
    assert rec.config["preprocessing"]["fps"] == 1.0
    assert rec.config["preprocessing"]["frame_selection"] == "fps"
```

In `tests/wrapper/test_reconstructor_preprocess.py` line ~33, change
`"preprocessing": {"frame_selection": "uniform", "frame_proportion": 0.1, "min_frames": 1, "max_frames": 5},`
to
`"preprocessing": {"frame_selection": "uniform", "fps": 1.0, "min_frames": None, "max_frames": 5},`

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -p no:randomly -k "extract_frames or fills_defaults" -v`
Expected: FAIL — `_extract_frames` still has the old signature, so the positional calls land wrong and `calls` never matches.

- [ ] **Step 3: Rewrite `_extract_frames`**

In `collab_splats/wrapper/reconstructor.py`, change the signature at lines 65-72 to:

```python
def _extract_frames(
    input_path: Path,
    frames_zarr: Path,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
) -> int:
```

Change the directory-branch provenance (line 88) to record the rate slot as unused:

```python
        prov = {
            "video_path": str(input_path),
            "video_mtime": None,
            "method": "dir",
            "fps": None,
            "max_frames": max_frames,
        }
```

Replace the sampler branch at lines 101-120 with:

```python
    # Video — 'fps' samples at a constant wall-clock rate (band-bounded), 'uniform'
    # spreads exactly max_frames over the whole video, 'optical_flow' picks high-motion
    # frames. Each method gets only its own knobs; sample_frames rejects the others.
    if frame_selection == "fps":
        frame_arrays, records = sample_frames(
            str(input_path),
            method="fps",
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
    elif frame_selection == "uniform":
        frame_arrays, records = sample_frames(str(input_path), method="uniform", max_frames=max_frames)
    elif frame_selection == "optical_flow":
        frame_arrays, records = sample_frames(str(input_path), method="optical_flow", max_frames=max_frames)
    else:
        raise ValueError(
            "preprocessing.frame_selection must be 'fps', 'uniform' or 'optical_flow', "
            f"got {frame_selection!r}"
        )
    method = frame_selection
```

Replace the video provenance dict at lines 132-137 with:

```python
    prov = {
        "video_path": str(input_path),
        "video_mtime": input_path.stat().st_mtime,
        "method": method,
        "fps": fps,
        "max_frames": max_frames,
    }
```

- [ ] **Step 4: Update the config read**

In `collab_splats/wrapper/reconstructor.py` at lines 499-506, replace:

```python
        n_frames = _extract_frames(
            input_path=Path(self.config["input_path"]),
            frames_zarr=self.frames_zarr,
            frame_selection=pre_cfg["frame_selection"],
            frame_proportion=pre_cfg["frame_proportion"],
            min_frames=pre_cfg["min_frames"],
            max_frames=pre_cfg["max_frames"],
        )
```

with:

```python
        n_frames = _extract_frames(
            input_path=Path(self.config["input_path"]),
            frames_zarr=self.frames_zarr,
            frame_selection=pre_cfg["frame_selection"],
            fps=pre_cfg["fps"],
            min_frames=pre_cfg["min_frames"],
            max_frames=pre_cfg["max_frames"],
        )
```

Strict `[...]` access is deliberate — `base.yaml` is the sole default source, and `tests/wrapper/test_reconstructor.py` has a guard test that fails on inline `.get(key, default)` fallbacks in this file.

- [ ] **Step 5: Run the wrapper tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py -p no:randomly -v`
Expected: the `_extract_frames` tests PASS. `test_init_fills_defaults_from_base_yaml` still FAILS (`KeyError: 'fps'` or `min_frames == 150`) — `base.yaml` changes in Task 5.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py
git commit -m "feat(wrapper): dispatch frame_selection to fps/uniform/optical_flow" -- collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py
```

---

### Task 5: Configs and docs

**Files:**
- Modify: `configs/base.yaml:16-22`, `configs/loop_closure.yaml:18-29`, `configs/README.md:224-227`, `docs/source/tutorials/03_splats/configs/base.yaml`

- [ ] **Step 1: Update `configs/base.yaml`**

Replace the `preprocessing:` block (lines 16-22) with:

```yaml
preprocessing:
  # Output: <output_path>/frames.zarr — canonical decode-once keyframe store
  # (chunked images + selection records + provenance); no images/ JPG dir.
  #
  # Each method has ONE density knob. max_frames is the frame budget: the target
  # COUNT for 'uniform', a ceiling for 'fps' and 'optical_flow'.
  frame_selection: fps        # fps | uniform | optical_flow
  fps: 1.0                    # 'fps' only: samples/second. Fixes the baseline between
                              # consecutive frames regardless of video length — which a
                              # count-based knob cannot do.
  min_frames: null            # 'fps' only: floor on the resulting count. null = fps is
                              # honoured literally (a floor would make it a hint).
  max_frames: 300             # frame budget; vggt_omega OOMs above ~300 on a 44 GB GPU
```

`frame_proportion` is deleted: with `fps` available it controls no physical quantity, and with `min_frames: null` it decides the count for no video at all.

- [ ] **Step 2: Update `configs/loop_closure.yaml`**

Replace the `preprocessing:` block (lines 18-29) with:

```yaml
preprocessing:
  # fps, NOT optical_flow: LC chains submaps through their shared overlap frame, so it
  # needs steady frame-to-frame overlap. optical_flow picks high-motion, low-overlap
  # frames → wide baseline across the submap hinge → submaps fail to register →
  # misaligned scene. Diversity helps coverage but hurts LC alignment.
  frame_selection: fps
  fps: 4.0
  # min_frames is the floor fps alone cannot guarantee: RAISE it to actually feed more
  # frames, otherwise submaps just re-chunk the same 200 keyframes into windows (same
  # coverage, no benefit). RAM-bound, not GPU-bound: fat submaps stay resident (~50 GB at
  # 1000 frames). Keep max_frames <= ~1000 on the 50 GB box.
  #
  # OWED: the LC calibrations (submap_size, submap_overlap, lc_retrieval_threshold, the
  # per-backbone target layers) and the 4-backbone ATE numbers were all measured under
  # 'uniform' spacing. The mechanism is unaffected; the numbers need re-verification.
  min_frames: 300
  max_frames: 600
```

- [ ] **Step 3: Update `configs/README.md`**

Replace the row at line 224 and delete the row at line 225:

```markdown
| `preprocessing.frame_selection` | str | `fps` | Frame sampling: `fps`, `uniform`, or `optical_flow` |
| `preprocessing.fps` | float | `1.0` | `fps` method only: samples per second |
| `preprocessing.min_frames` | int\|null | `null` | `fps` method only: floor on the resulting count |
| `preprocessing.max_frames` | int\|null | `300` | Frame budget: the COUNT for `uniform`, a ceiling for `fps`/`optical_flow` (vggt_omega OOMs above ~300) |
```

Then insert this section immediately above `## Config key reference (base.yaml)` (line 218):

```markdown
## Choosing a frame sampler

Each method has exactly one density knob. `max_frames` is the frame budget — the
target count for `uniform`, a ceiling for the other two.

| `frame_selection` | Density knob | What it holds constant |
|---|---|---|
| `fps` | `preprocessing.fps` | Wall-clock interval between frames — so the baseline between consecutive frames is fixed regardless of how long the video is. Count floats. |
| `uniform` | `preprocessing.max_frames` | Frame count. Spacing floats with video length. |
| `optical_flow` | `min_disparity` (creator-level) | Inter-frame motion. Both count and spacing float. |

Prefer `fps` for reconstruction: registration quality depends on the baseline
between consecutive frames, and a count-based knob leaves that free to vary by an
order of magnitude between a 1-minute and a 20-minute video.

**The band.** `fps` yields a count that grows with video length, so
`[min_frames, max_frames]` bounds it. Outside the band the targets are re-spread
evenly across the **whole** video and the effective fps is logged at WARNING —
never truncated, which would hand the reconstructor a scene that stops halfway.

**`max_frames` still dominates on long video.** At `fps: 1.0` the default
`max_frames: 300` binds past ~5 minutes, and beyond that the spacing is whatever
300 frames over the whole video gives you. The cap is a measured GPU limit, not a
preference — `fps` cannot route around it.

**Passing a knob that belongs to another method raises `ValueError`** (e.g. `fps=`
with `frame_selection: uniform`). There is no silently-ignored knob.
```

- [ ] **Step 4: Update the tutorial Reconstructor config**

In `docs/source/tutorials/03_splats/configs/base.yaml`, delete the `frame_proportion:` line and add `fps: 1.0` directly under the `frame_selection:` line, keeping the file's existing indentation. Leave its `min_frames` value as it is.

Verify no other in-tree config still references the deleted key:

Run: `grep -rn "frame_proportion" configs/ docs/source/ collab_splats/wrapper/reconstructor.py`
Expected: no matches. (`collab_splats/wrapper/splatter.py` keeps its own `frame_proportion` — that is a separate `ns-process-data` schema on a different code path and is deliberately untouched.)

- [ ] **Step 5: Run the wrapper tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -p no:randomly -v`
Expected: PASS — including `test_init_fills_defaults_from_base_yaml`, which now reads `fps: 1.0`, `min_frames: None`, `frame_selection: "fps"` from the new `base.yaml`.

- [ ] **Step 6: Commit**

```bash
git add configs/base.yaml configs/loop_closure.yaml configs/README.md docs/source/tutorials/03_splats/configs/base.yaml
git commit -m "feat(configs): default to fps sampling, delete frame_proportion" -- configs/base.yaml configs/loop_closure.yaml configs/README.md docs/source/tutorials/03_splats/configs/base.yaml
```

---

### Task 6: Dashboard

**Files:**
- Modify: `collab_splats/dashboard/config.py:19-22`, `collab_splats/dashboard/app.py:53,164-166,225-231,243,248`, `collab_splats/dashboard/pipeline.py:55-63,346-355,399-406`

- [ ] **Step 1: Add `fps` to `RunConfig`**

In `collab_splats/dashboard/config.py`, replace lines 19-22:

```python
    # Frame sampling
    sampling_method: str = "uniform"  # "uniform" | "optical_flow"
    max_frames: int = 100
    min_disparity: float = 50.0  # optical_flow only
```

with:

```python
    # Frame sampling — one density knob per method; max_frames is the frame budget
    # (the COUNT for "uniform", a ceiling for "fps"/"optical_flow").
    sampling_method: str = "fps"  # "fps" | "uniform" | "optical_flow"
    fps: float = 1.0  # fps only — samples/second
    max_frames: int = 100
    min_disparity: float = 50.0  # optical_flow only
```

`RunConfig.from_yaml` passes `**data` straight into the dataclass, so an older
`run_config.yaml` without an `fps` key still loads and picks up the `1.0` default.

- [ ] **Step 2: Add the widget and wire it up**

In `collab_splats/dashboard/app.py`:

Line 53 — add the new sampler, `fps` first to match the config default:

```python
_SAMPLERS = ["fps", "uniform", "optical_flow"]
```

After the `self.max_frames` widget (line 166), add:

```python
        # fps is consumed only by the fps sampler; hidden otherwise (see _bind_visibility below).
        self.fps = pn.widgets.FloatInput(name="fps (samples/sec)", value=s.get("fps", 1.0), start=0.01, step=0.5)
```

In the `self._persisted` dict (lines 225-231), add after the `"max_frames"` entry:

```python
            "fps": self.fps,
```

After the existing `min_disparity` visibility binding (line 243), add:

```python
        # fps is consumed only by the fps sampler; hide it otherwise.
        _bind_visibility(self.fps, self.sampling, lambda v: v == "fps")
```

In the sidebar `pn.Card` (line 248), add `self.fps` after `self.sampling`:

```python
            pn.Card(
                self.sampling, self.fps, self.max_frames, self.min_disparity, title="Frame sampling", collapsed=True
            ),
```

In `_current_config` (line ~429), add after `sampling_method=`:

```python
            fps=self.fps.value,
```

- [ ] **Step 3: Branch the pipeline sampler call**

In `collab_splats/dashboard/pipeline.py`, replace the body of `_sample` after the `op_log.update_progress(5, ...)` line (lines 346-355) with:

```python
    op_log.update_progress(5, f"sampling: {config.sampling_method}")
    # One knob per method — sample_frames rejects a knob belonging to another method,
    # so each branch passes only its own.
    method = config.sampling_method
    if method == "fps":
        return sample_frames(
            str(video_path),
            method="fps",
            fps=config.fps,
            max_frames=config.max_frames,
            on_progress=on_progress,
        )
    if method == "optical_flow":
        return sample_frames(
            str(video_path),
            method="optical_flow",
            min_disparity=config.min_disparity,
            max_frames=config.max_frames,
            on_progress=on_progress,
        )
    return sample_frames(
        str(video_path),
        method="uniform",
        max_frames=config.max_frames,
        on_progress=on_progress,
    )
```

- [ ] **Step 4: Record `fps` in the dashboard's frames.zarr provenance**

In `collab_splats/dashboard/pipeline.py`, replace `_write_frames_zarr` (lines 55-65) with:

```python
def _write_frames_zarr(
    frames: list[np.ndarray],
    records: list[dict],
    path: Path,
    *,
    video_path: Path,
    method: str,
    fps: float | None,
    max_frames: int,
) -> None:
    """Write the canonical frames.zarr (FrameStore schema) — feeds CameraLocalizer.from_feedforward
    so pixel reads bypass ff.image_paths (which may point at a directory this session doesn't own)."""
    prov = {
        "video_path": str(video_path),
        "video_mtime": Path(video_path).stat().st_mtime,
        "method": method,
        "fps": fps,
        "max_frames": max_frames,
    }
    FrameStore.create(path, frames, records, provenance=prov)
```

Then in `run_pipeline` (lines 399-406), replace the two-way method coercion and the call with:

```python
            sampling_method = config.sampling_method
            _write_frames_zarr(
                frames,
                records,
                out_dir / "frames.zarr",
                video_path=video_path,
                method=sampling_method,
                fps=config.fps if sampling_method == "fps" else None,
                max_frames=config.max_frames,
            )
```

- [ ] **Step 5: Run the dashboard tests and the mandatory smoke gate**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -p no:randomly -v`
Expected: PASS

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: prints `SMOKE PASS`. This gate is mandatory before any dashboard commit — do not commit if it fails.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/config.py collab_splats/dashboard/app.py collab_splats/dashboard/pipeline.py
git commit -m "feat(dashboard): expose the fps sampler with a rate input" -- collab_splats/dashboard/config.py collab_splats/dashboard/app.py collab_splats/dashboard/pipeline.py
```

---

### Task 7: Full-suite verification and CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (In-Flight Work section)

- [ ] **Step 1: Run the full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly -q`
Expected: PASS, with only the failures already listed in `docs/known-test-failures.md`.

If a large number (~60) of unrelated tests fail, suspect a concurrent session editing `configs/base.yaml` mid-run rather than a real regression — re-run once before investigating.

- [ ] **Step 2: Format**

Run: `/opt/venv/reconstruction/bin/python -m black collab_splats/preproc/sampling.py collab_splats/wrapper/reconstructor.py collab_splats/dashboard/config.py collab_splats/dashboard/app.py collab_splats/dashboard/pipeline.py tests/preproc/test_sampling.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py && /opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/sampling.py collab_splats/wrapper/reconstructor.py collab_splats/dashboard/config.py collab_splats/dashboard/app.py collab_splats/dashboard/pipeline.py tests/preproc/test_sampling.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py`

Format only the files this plan touched. **Never run repo-wide `black .`** — the venv's black is newer than the repo's formatting and would reformat unrelated files.

- [ ] **Step 3: Verify the end-to-end default path on the committed tutorial video**

Run:

```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.preproc import sample_frames, get_video_info
v = 'data/tutorial/tutorial.MP4'
import glob, pathlib
v = sorted(glob.glob('data/tutorial/*.MP4') + glob.glob('data/tutorial/*.mp4'))[0]
info = get_video_info(v)
print('video:', v, info['total_frames'], 'frames @', round(info['fps'], 2), 'fps')
f, r = sample_frames(v, method='fps', fps=1.0, min_frames=None, max_frames=300)
print('selected', len(f), 'first', r[0]['frame_idx'], 'last', r[-1]['frame_idx'], 'of', info['total_frames'])
assert len(f) == len(r)
assert r[-1]['frame_idx'] > info['total_frames'] * 0.8, 'coverage must reach the end of the video'
print('OK')
"
```

Expected: prints the frame count and `OK`. The final assertion is the real-video version of the truncation regression test — sampling must reach the end of the scene.

- [ ] **Step 4: Update CLAUDE.md**

In `/workspace/collab-splats/CLAUDE.md`, add to the "Recently completed" entries, immediately above the `mesh-tsdf-adapter-convergence` paragraph:

```markdown
Recently completed (2026-08-13): **fps-frame-sampling** — `frame_selection` gained an `fps` method that samples at a constant wall-clock rate (`_sample_fps` + `_fps_targets`), joining `uniform`/`optical_flow` as a third sibling over a shared `_sample_positions` body (single ffmpeg `select=` pass — the 14fdc42 fast path is untouched). `max_frames` now has ONE meaning everywhere — the frame budget: the target COUNT for `uniform`, a ceiling for `fps`/`optical_flow`. `min_frames` is an `fps`-only floor; outside `[min_frames, max_frames]` fps targets are **re-spread over the whole video, never truncated** (the old `targets[:max_frames]` dropped the tail). `frame_proportion` **deleted** — it controlled no physical quantity. Passing another method's knob raises `ValueError`. base.yaml defaults: `frame_selection: fps`, `fps: 1.0`, `min_frames: null`, `max_frames: 300`. **Owed: LC calibrations and the 4-backbone ATE numbers were measured under `uniform` spacing; `loop_closure.yaml` now uses `fps: 4.0` + `min_frames: 300` and the numbers need re-verification.** `splatter.py` keeps its own unrelated `frame_proportion`/`Literal["fps", ...]` on the nerfstudio path — deliberately untouched, names now collide ([spec](docs/superpowers/specs/2026-08-13-fps-frame-sampling-design.md) · [plan](docs/superpowers/plans/2026-08-13-fps-frame-sampling.md)).
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md collab_splats/ tests/
git commit -m "docs(preproc): record the fps sampling default and the owed LC re-verification" -- CLAUDE.md collab_splats/ tests/
```

---

## Verification Checklist

- [ ] `frame_selection: fps` with `fps: 1.0` is the shipping default in `configs/base.yaml`
- [ ] `frame_proportion` appears nowhere except `collab_splats/wrapper/splatter.py` (separate nerfstudio schema)
- [ ] Over-ceiling fps sampling reaches the END of the video (not `targets[:max_frames]`)
- [ ] Under-floor fps sampling re-spreads to `min_frames`
- [ ] A bound band logs the effective fps at WARNING
- [ ] `sample_frames` raises on `fps=` with a non-fps method, on `min_frames=` with a non-fps method, on `method="fps"` without `fps=`, and on `method="uniform"` without `max_frames=`
- [ ] The hidden `2.0` fps literal is gone from `sampling.py`
- [ ] `_iter_selected_frames` is still the only decode path for both target-list methods (one ffmpeg pass — `test_uniform_selects_in_one_pass_not_whole_video` still passes)
- [ ] `frames.zarr` provenance records `fps` from both `Reconstructor` and the dashboard
- [ ] `python -m collab_splats.dashboard --smoke` prints `SMOKE PASS`
- [ ] Full suite green except `docs/known-test-failures.md`
