# Affine Depth Alignment, Contiguous VDA Context, and 2DGS Surface Levers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise 2DGS mesh fidelity on the InstantSfM path by improving the depth targets (affine alignment in disparity, contiguous 8 FPS VDA context, evidence-bounded masking) and by adding a RaDe-GS median-normal term, with every change defaulting to today's behaviour.

**Architecture:** Six independent components landing back-to-front along the pipeline. `preproc/video.py` gains a context-frame grid and a chunked decode; `preproc/sampling.py` learns to draw keyframes only from that grid; `pointcloud/sfm.py` gains an affine disparity alignment (plus a one-sided far-extrapolation mask), a `keep_rows` filter on VDA writes, and a `random_seed`; `splats/{rendering,losses,trainer,outputs}.py` carry the 2DGS median depth through to a blended normal-consistency loss; `mesh/utils.py` lets TSDF fuse either rendered depth. Every new knob is off by default (`vda_context_fps: null`, `depth_align: scale`, `depth_ratio: 0.0`, `splat_depth: expected`), so an unchanged config reproduces the recorded baseline exactly.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), numpy, OpenCV, ffmpeg, zarr 3.x, pycolmap, torch, gsplat, pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-depth-align-affine-2dgs-surface-design.md`

---

## Conventions For Every Task

- Python is `/opt/venv/reconstruction/bin/python`. Never bare `python`.
- Format only the files you touched: `/opt/venv/reconstruction/bin/python -m black --target-version py311 <files> && /opt/venv/reconstruction/bin/python -m isort <files>`. Never run repo-wide.
- **Never commit foreign uncommitted work.** These carry other sessions' diffs: `CLAUDE.md`, `collab_splats/remote/rerun.py`, `configs/base.yaml`, `data/tutorial/README.md`, `docs/examples/run_pipeline_remote.py`, `pyproject.toml`, `tests/examples/test_run_pipeline_remote.py`, `tests/wrapper/test_reconstructor.py`, `uv.lock`. When a task touches one of these, stage with `git commit --only <paths>` and never `git add -A`.
- `docs/superpowers/` is gitignored — use `git add -f` for files under it.
- Code style: block comments above each logical run, one-line docstrings opening on the line after `"""`, `logging` not `print`.

---

## File Structure

| File | Responsibility | Component |
|---|---|---|
| `collab_splats/preproc/video.py` | `context_indices` (frame grid) + `decode_context` (chunked undistort/downscale decode) | A |
| `collab_splats/preproc/sampling.py` | `_sample_by_quality(candidates=)`, threaded through `sample_fps` / `sample_uniform` | A |
| `collab_splats/pointcloud/sfm.py` | `generate_vda_depth(keep_rows=)`, `_depth_correspondences`, `align_depth_affine`, `apply_depth_alignment(model=)`, `InstantSfMCreator.random_seed` | A, B, D, F |
| `collab_splats/wrapper/reconstructor.py` | wiring: `vda_context_fps`, `depth_align`, `random_seed`, `splat_depth`, honest sfm masking log | A, B, D, E, F |
| `collab_splats/splats/rendering.py` | keep the 2DGS median depth, emit `median_depth` + `depth_normal_median` | C |
| `collab_splats/splats/losses.py` | uniform `spec` 5th argument; blended normal consistency | C |
| `collab_splats/splats/trainer.py` | `depth_ratio` validation (range, 2dgs-only) | C |
| `collab_splats/splats/outputs.py` | write `median_depth` to `splats.zarr` for 2dgs | C |
| `collab_splats/mesh/utils.py` | `_splats_to_tsdf_inputs(depth_name=)` | E |
| `configs/base.yaml` | the four new knobs, documented | all |

---

## Task 1: Context frame grid (`context_indices`)

**Files:**
- Modify: `collab_splats/preproc/video.py` (append after `iter_frames`)
- Test: `tests/preproc/test_video.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/preproc/test_video.py`:

```python
def test_context_indices_matches_sample_fps_stride(tiny_video):
    # tiny_video is 60 frames @ 30 fps -> fps=10 gives stride 3
    grid = context_indices(tiny_video, target_fps=10.0)
    assert grid[:4] == [0, 3, 6, 9]
    assert grid[-1] < 60
    assert len(grid) == 20


def test_context_indices_floors_stride_at_one(tiny_video):
    # A target rate above the source rate cannot sample sub-frame
    grid = context_indices(tiny_video, target_fps=1000.0)
    assert grid == list(range(60))


def test_context_indices_reuses_a_probe(tiny_video):
    info = {"total_frames": 10, "fps": 30.0, "width": 320, "height": 240}
    assert context_indices(tiny_video, target_fps=15.0, info=info) == [0, 2, 4, 6, 8]
```

Add `context_indices` to the `from collab_splats.preproc.video import ...` line at the top of that test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k context_indices -v`
Expected: FAIL with `ImportError: cannot import name 'context_indices'`

- [ ] **Step 3: Write minimal implementation**

Append to `collab_splats/preproc/video.py`:

```python
def context_indices(video_path: str | Path, *, target_fps: float, info: dict | None = None) -> list[int]:
    """
    Source frame indices on a constant-rate grid at target_fps.

    - Uses the same `step = round(native_fps / target_fps)` rule as
      `sampling.sample_fps`, so a keyframe grid and a context grid built at the
      same rate agree frame-for-frame and keyframes are a subset by construction.
    - Stride floors at 1: a rate above the source rate cannot sample sub-frame.
    """
    if target_fps is None or target_fps <= 0:
        raise ValueError(f"context_indices needs a positive target_fps, got {target_fps!r}")

    info = info if info is not None else get_video_info(video_path)
    total = info["total_frames"]
    if total == 0:
        return []

    native_fps = info["fps"] or 30.0
    step = max(1, int(round(native_fps / target_fps)))
    return list(range(0, total, step))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k context_indices -v`
Expected: 3 passed

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/preproc/video.py tests/preproc/test_video.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/video.py tests/preproc/test_video.py
git commit --only collab_splats/preproc/video.py tests/preproc/test_video.py -m "feat(preproc): context_indices — constant-rate source frame grid"
```

---

## Task 2: Chunked context decode (`decode_context`)

**Files:**
- Modify: `collab_splats/preproc/video.py` (append after `context_indices`)
- Test: `tests/preproc/test_video.py`

Note on the spec: the design listed `decode_context(..., profile, roi, out_size)`. `roi` is dropped
here because `undistort_frames` derives it deterministically from the profile
(`cv2.getOptimalNewCameraMatrix`), so passing it in would let a caller silently disagree with the
crop the keyframes actually got. Everything else is as specced.

- [ ] **Step 1: Write the failing test**

Append to `tests/preproc/test_video.py`:

```python
def test_decode_context_downscales_to_short_side(tiny_video):
    # 320x240 source; short side 240 -> requested 120 halves both dimensions
    frames = decode_context(tiny_video, [0, 3, 6], out_short_side=120)
    assert frames.shape == (3, 120, 160, 3)
    assert frames.dtype == np.uint8


def test_decode_context_is_rgb_not_bgr(tiny_video):
    # tiny_video paints a pure-green rectangle; RGB output puts the peak in channel 1
    frames = decode_context(tiny_video, [30], out_short_side=240)
    patch = frames[0, 90:110, 130:150]
    assert patch[..., 1].mean() > patch[..., 0].mean()
    assert patch[..., 1].mean() > patch[..., 2].mean()


def test_decode_context_chunking_does_not_change_output(tiny_video):
    indices = [0, 3, 6, 9, 12]
    one_chunk = decode_context(tiny_video, indices, out_short_side=120, chunk_size=64)
    many_chunks = decode_context(tiny_video, indices, out_short_side=120, chunk_size=2)
    np.testing.assert_array_equal(one_chunk, many_chunks)


def test_decode_context_empty_indices(tiny_video):
    frames = decode_context(tiny_video, [], out_short_side=120)
    assert frames.shape[0] == 0


def test_decode_context_undistorts_before_downscaling(tiny_video):
    # A profile calibrated at the SOURCE resolution: undistortion must happen at
    # 320x240, so it must not raise, and the alpha=0 crop shrinks the frame.
    profile = DistortionProfile(width=320, height=240, fx=300.0, fy=300.0, cx=160.0, cy=120.0,
                                k1=-0.2, k2=0.0, p1=0.0, p2=0.0)
    frames = decode_context(tiny_video, [0, 3], profile=profile, out_short_side=100)
    assert frames.shape[0] == 2
    assert min(frames.shape[1:3]) == 100
```

Add to that file's imports:

```python
from collab_splats.preproc.undistort import DistortionProfile
from collab_splats.preproc.video import context_indices, decode_context
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k decode_context -v`
Expected: FAIL with `ImportError: cannot import name 'decode_context'`

- [ ] **Step 3: Write minimal implementation**

Append to `collab_splats/preproc/video.py`:

```python
def decode_context(
    video_path: str | Path,
    indices: Sequence[int],
    *,
    profile=None,
    out_short_side: int = 518,
    chunk_size: int = 64,
) -> np.ndarray:
    """
    Decode a context frame grid as RGB, undistorted at native resolution, then downscaled.

    - `indices` come from `context_indices`; the return is (N, h, w, 3) uint8 RGB in that order.
    - `profile` is the DistortionProfile frames.zarr was written with (None = raw frames).
      Undistortion runs at the SOURCE resolution before any downscale, or the alpha=0 crop
      and K_new stop matching the keyframes.
    - `out_short_side` is the model's native grid (VDA resizes the short side to 518 and
      upscales anything smaller, so decoding below that loses detail without saving GPU).
    - Chunked so a 2000-frame grid never holds 2000 full-resolution frames at once.
    """
    # Lazy import: undistort pulls pycolmap, which video.py otherwise never needs
    from collab_splats.preproc.undistort import undistort_frames

    ordered = sorted({int(i) for i in indices})
    if not ordered:
        return np.zeros((0, out_short_side, out_short_side, 3), dtype=np.uint8)

    out: list[np.ndarray] = []
    for start in range(0, len(ordered), chunk_size):
        chunk = ordered[start : start + chunk_size]

        # One ffmpeg select pass per chunk, BGR at source resolution
        decoded = dict(iter_frames(video_path, indices=chunk))
        bgr_frames = [decoded[i] for i in chunk if i in decoded]
        if not bgr_frames:
            continue

        # Undistort at native resolution — the crop is what makes the context aspect
        # ratio match the keyframes'
        if profile is not None:
            bgr_frames, _K_new, _roi = undistort_frames(bgr_frames, profile)

        # Downscale to the model grid (INTER_AREA: this is always a shrink), then BGR -> RGB
        height, width = bgr_frames[0].shape[:2]
        scale = out_short_side / min(height, width)
        target = (int(round(width * scale)), int(round(height * scale)))
        for bgr in bgr_frames:
            small = cv2.resize(bgr, target, interpolation=cv2.INTER_AREA)
            out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    return np.stack(out).astype(np.uint8)
```

Ensure `import cv2` and `from collections.abc import Sequence` are present at the top of `video.py` (`Sequence` already is — `iter_frames` uses it). Add `import cv2` if absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -v`
Expected: all pass

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/preproc/video.py tests/preproc/test_video.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/video.py tests/preproc/test_video.py
git commit --only collab_splats/preproc/video.py tests/preproc/test_video.py -m "feat(preproc): decode_context — chunked undistort-then-downscale context decode"
```

---

## Task 3: Restrict keyframe selection to a candidate grid

**Files:**
- Modify: `collab_splats/preproc/sampling.py:237-296` (`_sample_by_quality`), `:298-333` (`sample_uniform`), `:335-400` (`sample_fps`)
- Test: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/preproc/test_sampling.py`:

```python
def test_candidates_restrict_chosen_frames_to_the_grid(tiny_video, clean_report):
    # 60-frame video, grid every 3rd frame, 10 keyframes -> every pick is a grid member
    grid = list(range(0, 60, 3))
    _frames, records = sample_uniform(
        tiny_video, max_frames=10, report=clean_report, search_radius=7, candidates=grid
    )
    chosen = [r["frame_idx"] for r in records]
    assert set(chosen) <= set(grid)
    assert len(chosen) == len(set(chosen))


def test_candidates_substitute_a_blurry_target_within_the_grid(tiny_video):
    # Grid every 3rd frame (20 members), 5 targets -> grid spacing 4 -> radius 1, so each
    # window is 3 grid members wide and substitution is actually possible. Target 30 is an
    # exact grid member and unusable, so the pick must move to 27 or 33 — never to 29 or 31.
    report = _synthetic_report(60, bad=(30,))
    grid = list(range(0, 60, 3))
    _frames, records = sample_uniform(
        tiny_video, max_frames=5, report=report, search_radius=7, candidates=grid
    )
    chosen = [r["frame_idx"] for r in records]
    assert 30 not in chosen
    assert {27, 33} & set(chosen)
    assert set(chosen) <= set(grid)


def test_candidates_none_is_byte_identical_to_today(tiny_video, clean_report):
    _f1, r1 = sample_uniform(tiny_video, max_frames=10, report=clean_report, search_radius=3)
    _f2, r2 = sample_uniform(
        tiny_video, max_frames=10, report=clean_report, search_radius=3, candidates=None
    )
    assert [r["frame_idx"] for r in r1] == [r["frame_idx"] for r in r2]


def test_sample_fps_accepts_candidates(tiny_video, clean_report):
    grid = list(range(0, 60, 3))
    _frames, records = sample_fps(
        tiny_video, fps=5.0, report=clean_report, search_radius=7, candidates=grid
    )
    assert set(r["frame_idx"] for r in records) <= set(grid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k candidates -v`
Expected: FAIL with `TypeError: sample_uniform() got an unexpected keyword argument 'candidates'`

- [ ] **Step 3: Write the implementation**

In `collab_splats/preproc/sampling.py`, change `_sample_by_quality`'s signature to add `candidates`:

```python
def _sample_by_quality(
    video_path: str,
    targets: list[int],
    *,
    total: int,
    report: dict,
    quality: dict | None,
    search_radius: int,
    on_progress,
    desc: str,
    candidates: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
```

Extend its docstring with one bullet:

```
    - candidates restricts BOTH the target and its substitutes to a fixed index grid
      (the VDA context grid), so every keyframe is a grid member by construction. The
      window radius is then counted in grid steps, not source frames.
```

Replace the window/choose block (the `spacing`/`radius`/`chosen` section) with:

```python
    # Window radius: half the target spacing, capped. This does NOT guarantee that
    # neighbouring windows never overlap — the no-grid branch derives spacing from the
    # FIRST target gap only, and grid mode's spacing is an average — so the dedup pass
    # below is what actually keeps the index map deterministic.
    if candidates is not None:
        # Grid mode: spacing and radius are counted in grid steps, and each target snaps
        # to the first grid member at or after it (searchsorted side="left", so a target
        # that is already a grid member maps to itself) before the window is cut.
        grid = np.asarray(sorted({int(c) for c in candidates}), dtype=np.int64)
        if grid.size == 0:
            raise ValueError("_sample_by_quality: candidates is empty")
        spacing = grid.size / len(targets) if len(targets) > 1 else grid.size
        radius = min(max(int((spacing - 1) // 2), 0), search_radius)
        positions = np.clip(np.searchsorted(grid, targets), 0, grid.size - 1)
        windows = [grid[max(0, p - radius) : p + radius + 1].tolist() for p in positions]
    else:
        spacing = targets[1] - targets[0] if len(targets) > 1 else total
        radius = min(max((spacing - 1) // 2, 0), search_radius)
        windows = [sorted({min(max(t + o, 0), total - 1) for o in range(-radius, radius + 1)}) for t in targets]

    # Per target, prefer a usable frame and break ties on sharpness. max() over an
    # ascending range returns the FIRST maximal element, matching the old strict
    # `key > best` comparison — the tie-break is parity-critical.
    chosen: list[int] = [max(window, key=lambda i: (bool(usable[i]), float(laplacian[i]))) for window in windows]

    # Two targets can snap to one grid member when the grid is coarse relative to the
    # budget; keep the first and say so rather than writing a duplicate frame.
    deduped = list(dict.fromkeys(chosen))
    if len(deduped) != len(chosen):
        logger.warning(
            "%d of %d targets collapsed onto an already-chosen frame (candidate grid too coarse "
            "for the frame budget); keeping %d unique frames",
            len(chosen) - len(deduped), len(chosen), len(deduped),
        )
        chosen = deduped
```

Add `candidates: Sequence[int] | None = None` as the last keyword parameter of both `sample_uniform` and `sample_fps`, pass `candidates=candidates` through in each `_sample_by_quality(...)` call, and add one docstring bullet to each:

```
    - candidates: restrict every selected frame to this index grid (see _sample_by_quality).
```

Confirm `Sequence` is imported in `sampling.py`; if not, add `from collections.abc import Callable, Sequence`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v`
Expected: all pass, including the pre-existing `tests/preproc/test_sampling_parity.py`

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit --only collab_splats/preproc/sampling.py tests/preproc/test_sampling.py -m "feat(preproc): draw keyframes and blur substitutes from a candidate grid"
```

---

## Task 4: `generate_vda_depth(keep_rows=)` + correct the metric-head comment

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py:242-330`
- Test: `tests/pointcloud/test_instantsfm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/pointcloud/test_instantsfm.py`:

```python
def test_keep_rows_length_must_match_names(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((6, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="keep_rows"):
        sfm.generate_vda_depth(frames, fps=8.0, out_dir=tmp_path, names=_NAMES, keep_rows=[0, 2, 4])


def test_keep_rows_must_be_in_range(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="keep_rows"):
        sfm.generate_vda_depth(frames, fps=8.0, out_dir=tmp_path, names=_NAMES, keep_rows=[0, 9])


def test_keep_rows_writes_only_the_requested_rows(tmp_path, monkeypatch):
    # Stub inference: 5 context frames in, one distinguishable depth map per frame
    def _fake_model(**_kwargs):
        class _M:
            def load_state_dict(self, *_a, **_k):
                return None

            def to(self, *_a, **_k):
                return self

            def eval(self):
                return self

            def infer_video_depth(self, frames, fps, **_k):
                maps = np.stack([np.full((8, 8), float(i + 1), dtype=np.float32) for i in range(len(frames))])
                return maps, fps

        return _M()

    monkeypatch.setattr(sfm, "_load_vda_model", _fake_model)
    (tmp_path / "ckpt").mkdir()
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    sfm.generate_vda_depth(
        frames, fps=8.0, out_dir=tmp_path, names=_NAMES, keep_rows=[1, 3], depth_width=8
    )

    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    assert sorted(p.name for p in npy_dir.iterdir()) == ["frame_000000.npy", "frame_000001.npy"]
    assert np.load(npy_dir / "frame_000000.npy").flat[0] == pytest.approx(2.0)
    assert np.load(npy_dir / "frame_000001.npy").flat[0] == pytest.approx(4.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_instantsfm.py -k keep_rows -v`
Expected: FAIL with `TypeError: generate_vda_depth() got an unexpected keyword argument 'keep_rows'`

- [ ] **Step 3: Write the implementation**

In `collab_splats/pointcloud/sfm.py`, extract model construction so tests can stub it. Insert above `generate_vda_depth`:

```python
def _load_vda_model(*, encoder: str, device: str):
    """
    Construct the VDA metric model on `device` from the pinned checkpoint.

    - Split out from generate_vda_depth so the write path is testable without a GPU or
      the third_party clone.
    """
    # Lazy heavy import — VDA lives in a third_party clone (repo root on sys.path), not
    # site-packages. Upstream HEAD (4f5ae23) has no metric_depth/ subdir: `video_depth_anything/`
    # sits at the clone root and `video_depth.py:27` imports a TOP-LEVEL `utils` namespace
    # package (`utils/util.py`) from the same root. Probed 2026-08-23: no foreign top-level
    # `utils` in the venv, and importing collab_splats.wrapper.reconstructor leaves none in
    # sys.modules — a regular `utils` package anywhere on sys.path would shadow VDA's namespace
    # one regardless of insert order, so re-probe if a dependency ever ships one.
    if not (VDA_ROOT / "video_depth_anything").is_dir():
        raise ImportError(
            f"Video-Depth-Anything clone not found at {VDA_ROOT} — run setup.sh "
            "(clones the repo at 4f5ae23 and downloads the metric vitl checkpoint)"
        )
    if str(VDA_ROOT) not in sys.path:
        sys.path.insert(0, str(VDA_ROOT))
    from video_depth_anything.video_depth import VideoDepthAnything

    ckpt = VDA_ROOT / "checkpoints" / VDA_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"VDA metric checkpoint missing: {ckpt} — run setup.sh")

    # metric=True loads the metric head AND disables infer_video_depth's cross-window
    # scale-and-shift chaining (video_depth.py:135), so consecutive windows are stitched
    # on the head's own absolute output rather than fitted to each other. Measured
    # 2026-08-26: this is why a full-video pass does not improve metric contiguity.
    model = VideoDepthAnything(**_VDA_MODEL_CONFIGS[encoder], metric=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.to(device).eval()
```

Replace `generate_vda_depth`'s signature and the parts of its body that changed:

```python
def generate_vda_depth(
    frames: np.ndarray,
    fps: float,
    out_dir: Path,
    names: list[str],
    *,
    encoder: str = "vitl",
    input_size: int = 518,
    depth_width: int = 518,
    device: str = "cuda",
    keep_rows: Sequence[int] | None = None,
) -> Path:
```

Add two docstring bullets after the `frames:` bullet:

```
    - keep_rows: when set, `frames` is a CONTEXT stream (a contiguous constant-rate grid)
      and only these rows are written, one per entry of `names`, in order. VDA is temporal,
      so inference sees the whole stream and only the write is filtered.
```

Replace the length check at the top of the body with:

```python
    if keep_rows is None:
        if len(names) != len(frames):
            raise ValueError(f"names ({len(names)}) and frames ({len(frames)}) must align one-to-one")
    else:
        keep_rows = [int(r) for r in keep_rows]
        if len(names) != len(keep_rows):
            raise ValueError(f"names ({len(names)}) and keep_rows ({len(keep_rows)}) must align one-to-one")
        out_of_range = [r for r in keep_rows if not 0 <= r < len(frames)]
        if out_of_range:
            raise ValueError(
                f"keep_rows out of range for {len(frames)} context frames (first: {out_of_range[0]})"
            )
```

Replace the model-construction block (from the `# Lazy heavy import` comment through `model = model.to(device).eval()`) with:

```python
    model = _load_vda_model(encoder=encoder, device=device)
```

Replace the inference log line and the write loop:

```python
    # Metric inference over the whole sequence (returns input-res depth). With keep_rows
    # the stream is the context grid and `fps` is the CONTEXT rate, not the keyframe rate.
    logger.info(
        "VDA metric inference: %d frames @ %.2f fps (encoder=%s, writing %d maps)",
        len(frames), fps, encoder, len(names),
    )
    depths, _fps = model.infer_video_depth(frames, fps, input_size=input_size, device=device, fp32=False)
    depths = np.asarray(depths, dtype=np.float32)

    # Free the GPU before the caller's InstantSfM CUDA step — resize/write below is CPU-only
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Keep only the rows the caller asked for (all of them when keep_rows is None)
    if keep_rows is not None:
        depths = depths[np.asarray(keep_rows, dtype=np.int64)]

    # Nearest-resize to depth_width and write one map per frame, keyed by image stem
    h, w = depths.shape[1:3]
    depth_hw = (int(round(depth_width * h / w)), depth_width)
    npy_dir.mkdir(parents=True, exist_ok=True)
    for name, depth in zip(names, depths):
        small = cv2.resize(depth, (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST)
        np.save(npy_dir / f"{Path(name).stem}.npy", small.astype(np.float32))
    logger.info("VDA depths written: %s (%d maps @ %dx%d)", npy_dir, len(names), depth_hw[1], depth_hw[0])
    return depth_dir
```

Confirm `from collections.abc import Sequence` is imported at the top of `sfm.py`; add it if not.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_instantsfm.py -v`
Expected: all pass (the pre-existing `test_vda_missing_clone_raises_actionable_import_error` still passes — `_load_vda_model` raises the same `ImportError`)

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/pointcloud/sfm.py tests/pointcloud/test_instantsfm.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/pointcloud/sfm.py tests/pointcloud/test_instantsfm.py
git commit --only collab_splats/pointcloud/sfm.py tests/pointcloud/test_instantsfm.py -m "feat(sfm): generate_vda_depth keep_rows for context-stream inference"
```

---

## Task 5: Affine-in-disparity alignment with a one-sided far bound

Two commits: the pure refactor that exposes the correspondences, then the alignment model that consumes them.

### Part A: Extract `_depth_correspondences` (pure refactor)

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py:340-430`
- Test: `tests/pointcloud/test_depth_align.py` (create)

- [ ] **Step 1: Write the characterisation test**

Create `tests/pointcloud/test_depth_align.py`:

```python
"""
Depth alignment: track-observation correspondences, scale fit, affine-in-disparity fit.
"""

import numpy as np
import pytest

from collab_splats.pointcloud import sfm

GRID_H, GRID_W = 16, 32
CAM_W, CAM_H = 64, 32


def _fake_reconstruction(per_image_depths):
    """
    Duck-typed pycolmap stand-in: one image per entry, observations at fixed pixels.

    - per_image_depths maps "frame_NNNNNN.jpg" -> list of (u_grid, v_grid, d_colmap).
      Pixels are given in DEPTH-GRID coordinates and scaled up to camera resolution here,
      so a test states where in the depth map an observation lands.
    """

    class _Camera:
        def __init__(self):
            self.width, self.height = CAM_W, CAM_H

    class _Point2D:
        def __init__(self, xy, point3D_id):
            self.xy = np.asarray(xy, dtype=np.float64)
            self.point3D_id = point3D_id

        def has_point3D(self):
            return self.point3D_id is not None

    class _Point3D:
        def __init__(self, xyz):
            self.xyz = np.asarray(xyz, dtype=np.float64)

    class _Image:
        def __init__(self, name, points2D):
            self.name = name
            self.camera_id = 1
            self.points2D = points2D

        def cam_from_world(self):
            # Identity pose: a point's world xyz IS its camera-frame xyz, so xyz[2] = d_colmap
            class _Pose:
                def matrix(self_inner):
                    return np.eye(4)

            return _Pose()

    points3D, images = {}, {}
    next_id = 1
    for image_id, (name, observations) in enumerate(per_image_depths.items(), start=1):
        points2D = []
        for u_grid, v_grid, d_colmap in observations:
            points3D[next_id] = _Point3D([0.0, 0.0, d_colmap])
            points2D.append(
                _Point2D([u_grid * CAM_W / GRID_W, v_grid * CAM_H / GRID_H], next_id)
            )
            next_id += 1
        images[image_id] = _Image(name, points2D)

    class _Recon:
        pass

    recon = _Recon()
    recon.points3D = points3D
    recon.images = images
    recon.cameras = {1: _Camera()}
    return recon


def test_correspondences_pair_track_depth_with_sampled_vda_depth():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0), (7, 8, 20.0)]})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    depth[0, 4, 3] = 5.0
    depth[0, 8, 7] = 8.0

    pairs = sfm._depth_correspondences(recon, ["frame_000000.jpg"], depth)
    assert len(pairs) == 1
    d_colmap, d_vda = pairs[0]
    np.testing.assert_allclose(sorted(d_colmap), [10.0, 20.0])
    np.testing.assert_allclose(sorted(d_vda), [5.0, 8.0])


def test_correspondences_drop_zero_and_out_of_bounds_samples():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0), (5, 5, 20.0)]})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    depth[0, 4, 3] = 5.0  # (5,5) is left at 0 -> dropped

    d_colmap, d_vda = sfm._depth_correspondences(recon, ["frame_000000.jpg"], depth)[0]
    assert len(d_colmap) == 1


def test_correspondences_raise_on_unregistered_name():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0)]})
    depth = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)
    with pytest.raises(ValueError, match="not in reconstruction"):
        sfm._depth_correspondences(recon, ["frame_000000.jpg", "frame_000009.jpg"], depth)


def test_scale_alignment_recovers_a_constant_ratio():
    # 30 observations at exactly 2x -> the frame's fitted scale is 2.0
    observations = [(i % GRID_W, i % GRID_H, 2.0 * (i + 1)) for i in range(30)]
    recon = _fake_reconstruction({"frame_000000.jpg": observations})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(30):
        depth[0, i % GRID_H, i % GRID_W] = float(i + 1)

    scales, stats = sfm.align_depth_to_reconstruction(recon, ["frame_000000.jpg"], depth)
    assert scales[0] == pytest.approx(2.0, rel=1e-6)
    assert stats["n_fallback"] == 0
```

Note: the observation pixels above collide in the depth map for `i >= 16`; that is fine — every
colliding pair still satisfies `d_colmap = 2 * d_vda`, which is what the test asserts.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_align.py -v`
Expected: FAIL with `AttributeError: module 'collab_splats.pointcloud.sfm' has no attribute '_depth_correspondences'`

- [ ] **Step 3: Write the implementation**

In `collab_splats/pointcloud/sfm.py`, insert above `align_depth_to_reconstruction`:

```python
def _depth_correspondences(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Per-frame (d_colmap, d_vda) pairs from track observations, both positive and in bounds.

    - Each points2D carrying a point3D gives an exact pixel plus that point's z in the camera
      frame; the pixel is rescaled from native camera resolution to the depth grid and
      nearest-sampled into VDA depth.
    - Returns one (d_colmap, d_vda) tuple per row of `depth`, in `image_names` order; a frame
      with no usable observation gets a pair of empty arrays.
    """
    # Row order is the caller's; every name must be registered
    name_to_image = {image.name: image for image in reconstruction.images.values()}
    missing = [name for name in image_names if name not in name_to_image]
    if missing:
        raise ValueError(f"{len(missing)} image names not in reconstruction (first: {missing[0]})")

    _n_frames, grid_h, grid_w = depth.shape
    empty = (np.zeros(0), np.zeros(0))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for row, name in enumerate(image_names):
        image = name_to_image[name]
        camera = reconstruction.cameras[image.camera_id]

        # Track observations: exact 2D pixel + the observed point's depth in this view
        observations = [p for p in image.points2D if p.has_point3D()]
        if not observations:
            pairs.append(empty)
            continue
        xyz = np.stack([reconstruction.points3D[p.point3D_id].xyz for p in observations])
        cam_from_world = image.cam_from_world().matrix()
        d_colmap = (xyz @ cam_from_world[:3, :3].T + cam_from_world[:3, 3])[:, 2]

        # Rescale native pixels to the depth grid (the localization ref_px bug class —
        # native-res keypoints indexed into a model-res grid), then nearest-sample
        xy = np.stack([p.xy for p in observations])
        u = np.rint(xy[:, 0] * (grid_w / camera.width)).astype(np.int64)
        v = np.rint(xy[:, 1] * (grid_h / camera.height)).astype(np.int64)
        in_bounds = (u >= 0) & (u < grid_w) & (v >= 0) & (v < grid_h)
        d_vda = np.zeros(len(observations))
        d_vda[in_bounds] = depth[row, v[in_bounds], u[in_bounds]]

        # Keep pairs with positive depth on both sides
        valid = in_bounds & (d_vda > 0) & (d_colmap > 0)
        pairs.append((d_colmap[valid], d_vda[valid]))

    return pairs
```

Replace the whole per-frame loop inside `align_depth_to_reconstruction` (from `# Row order is the caller's` through the end of the `for row, name in enumerate(image_names):` block) with:

```python
    n_frames = depth.shape[0]
    scales = np.full(n_frames, np.nan)
    obs_counts = np.zeros(n_frames, dtype=np.int64)
    pooled_ratios: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []

    # One robust scale per frame from its track observations; below the obs floor, don't fit
    for row, (d_colmap, d_vda) in enumerate(_depth_correspondences(reconstruction, image_names, depth)):
        obs_counts[row] = len(d_colmap)
        if obs_counts[row] == 0:
            continue
        ratios = d_colmap / d_vda
        pooled_ratios.append(ratios)
        pooled_rows.append(np.full(len(ratios), row))
        if obs_counts[row] >= MIN_ALIGN_OBS:
            scales[row] = np.median(ratios)
```

The rest of `align_depth_to_reconstruction` (fallback handling, stats, return) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_align.py tests/pointcloud/test_instantsfm.py -v`
Expected: all pass

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py
git add -f tests/pointcloud/test_depth_align.py
git commit --only collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py -m "refactor(sfm): extract _depth_correspondences from align_depth_to_reconstruction"
```

---

### Part B: Affine-in-disparity alignment with a one-sided far bound

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py` (append after `align_depth_to_reconstruction`)
- Test: `tests/pointcloud/test_depth_align.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/pointcloud/test_depth_align.py`:

```python
def _affine_scene(a, b, n_obs=200, d_min=2.0, d_max=40.0):
    """
    One frame whose true mapping is 1/d_colmap = a*(1/d_vda) + b, sampled on the depth grid.
    """
    rng = np.random.default_rng(0)
    d_vda_values = rng.uniform(d_min, d_max, n_obs)
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    observations = []
    for i, d_vda in enumerate(d_vda_values):
        u, v = i % GRID_W, (i // GRID_W) % GRID_H
        depth[0, v, u] = d_vda
        observations.append((u, v, float(1.0 / (a / d_vda + b))))
    return _fake_reconstruction({"frame_000000.jpg": observations}), depth


def test_affine_recovers_the_generating_coefficients():
    recon, depth = _affine_scene(a=1.5, b=-0.004)
    coeffs, _far_limits, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert coeffs[0, 0] == pytest.approx(1.5, rel=1e-4)
    assert coeffs[0, 1] == pytest.approx(-0.004, abs=1e-6)
    assert stats["n_fallback"] == 0


def test_affine_is_robust_to_gross_outliers():
    recon, depth = _affine_scene(a=1.5, b=-0.004)
    # Corrupt 5% of the observations with a 100x depth error
    for point in list(recon.points3D.values())[:10]:
        point.xyz[2] *= 100.0

    coeffs, _far, _stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert coeffs[0, 0] == pytest.approx(1.5, rel=5e-3)
    assert coeffs[0, 1] == pytest.approx(-0.004, abs=5e-5)


def test_affine_falls_back_to_scale_below_the_obs_floor():
    # 30 observations: above MIN_ALIGN_OBS (20) but below MIN_AFFINE_OBS (50)
    observations = [(i % GRID_W, i % GRID_H, 2.0 * (i + 1)) for i in range(30)]
    recon = _fake_reconstruction({"frame_000000.jpg": observations})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(30):
        depth[0, i % GRID_H, i % GRID_W] = float(i + 1)

    coeffs, _far, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert coeffs[0, 1] == 0.0                       # scale-only: b is exactly zero
    assert coeffs[0, 0] == pytest.approx(0.5)        # a = 1/s with s = 2.0
    assert stats["n_fallback"] == 1


def test_affine_far_limit_is_the_furthest_fitted_observation():
    recon, depth = _affine_scene(a=1.0, b=0.0, d_min=2.0, d_max=40.0)
    _coeffs, far_limits, _stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    # Observations stop at 40 m, so nothing beyond ~40 m has evidence
    assert far_limits[0] == pytest.approx(40.0, rel=0.05)


def test_apply_affine_inverts_the_fitted_mapping():
    depth = np.array([[5.0, 10.0, 20.0]], dtype=np.float32)
    out = sfm._apply_affine_depth(depth, a=1.5, b=-0.004, far_limit=np.inf)
    expected = depth / (1.5 - 0.004 * depth)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_apply_affine_zeroes_saturated_and_far_pixels():
    depth = np.array([[5.0, 100.0, 400.0]], dtype=np.float32)
    # a + b*d goes non-positive at d = 250; far_limit cuts at 90
    out = sfm._apply_affine_depth(depth, a=1.0, b=-0.004, far_limit=90.0)
    assert out[0, 0] > 0.0
    assert out[0, 1] == 0.0   # beyond far_limit
    assert out[0, 2] == 0.0   # saturated AND beyond far_limit


def test_apply_affine_keeps_zeros_zero():
    depth = np.array([[0.0, 5.0]], dtype=np.float32)
    out = sfm._apply_affine_depth(depth, a=1.0, b=0.0, far_limit=np.inf)
    assert out[0, 0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_align.py -k affine -v`
Expected: FAIL with `AttributeError: module 'collab_splats.pointcloud.sfm' has no attribute 'align_depth_affine'`

- [ ] **Step 3: Write the implementation**

In `collab_splats/pointcloud/sfm.py`, add beside `MIN_ALIGN_OBS`:

```python
MIN_AFFINE_OBS = 50  # per-frame floor for the 2-parameter disparity fit (scale needs only 20)
AFFINE_REJECT_ROUNDS = 2  # MAD-3sigma rejection passes over the least-squares fit
AFFINE_EPS = 1e-9  # positivity floor on the fitted disparity at the frame's far end
```

Append after `align_depth_to_reconstruction`:

```python
def _fit_affine_disparity(d_colmap: np.ndarray, d_vda: np.ndarray) -> tuple[float, float] | None:
    """
    Least-squares 1/d_colmap ~= a*(1/d_vda) + b with two MAD-3sigma rejection rounds.

    - Disparity, not depth: gsplat's depth_l1_loss is L1 on 1/d, so the fit is done in the
      space the loss is paid in. Measured 2026-08-26: the offset term b carries the entire
      -18.9% loss-floor improvement over a scale-only fit.
    - Returns None when the surviving inlier set is too small or degenerate to solve.
    """
    q_vda = 1.0 / d_vda
    q_colmap = 1.0 / d_colmap
    inliers = np.ones(len(q_vda), dtype=bool)

    for _round in range(AFFINE_REJECT_ROUNDS + 1):
        if inliers.sum() < MIN_AFFINE_OBS:
            return None

        # Solve the 2-parameter normal equations over the current inliers
        design = np.stack([q_vda[inliers], np.ones(int(inliers.sum()))], axis=1)
        solution, _residuals, rank, _sv = np.linalg.lstsq(design, q_colmap[inliers], rcond=None)
        if rank < 2:
            return None
        a, b = float(solution[0]), float(solution[1])

        # Reject at 3 MAD (scaled to sigma) and refit; a zero MAD means an exact fit
        residual = q_colmap - (a * q_vda + b)
        mad = float(np.median(np.abs(residual[inliers] - np.median(residual[inliers]))))
        if mad <= 0:
            return a, b
        inliers = np.abs(residual) <= 3.0 * 1.4826 * mad

    return a, b


def _apply_affine_depth(depth_row: np.ndarray, a: float, b: float, far_limit: float) -> np.ndarray:
    """
    Map one VDA depth map through the fitted affine disparity: d_new = d / (a + b*d).

    - Applied in depth form so a zero-depth pixel never divides; zeros stay zero.
    - Pixels where the denominator collapses (b < 0 saturates past a horizon) and pixels
      beyond `far_limit` (no track evidence at that range) are written as 0 = no target.
    """
    denominator = a + b * depth_row
    out = np.zeros_like(depth_row, dtype=np.float32)
    supported = (depth_row > 0) & (denominator > AFFINE_EPS) & (depth_row <= far_limit)
    out[supported] = depth_row[supported] / denominator[supported]
    return out


def align_depth_affine(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Per-frame affine-in-disparity alignment of VDA depth to the reconstruction's world.

    - Returns (coeffs (N,2) [a, b], far_limits (N,) in INPUT VDA DEPTH UNITS, stats). Apply
      with `_apply_affine_depth`; a scale-only frame is returned as (1/s, 0.0), which is the
      same mapping the scale path applies.
    - Falls back to scale-only when a frame has fewer than MIN_AFFINE_OBS observations, the
      fit is unsolvable, a <= 0, or the fitted disparity is non-positive at the frame's far
      end. The far end is p99 of the depth map, not its max: a single sky pixel rejects
      78/300 frames, p99 rejects 14/300 (measured 2026-08-26).
    - far_limits bounds where the fit stops being interpolation. Beyond it the fit is
      extrapolating, and held-out observations there are 2.2x worse. The bound is ONE-SIDED
      by design: SIFT tracks do not cover close surfaces, so a near-side bound would delete
      the closest ~4% of every frame — the near-field geometry this exists to sharpen.

    AMENDED by Task 5's review remediation — the shipped contract is stricter than the
    code block below, which is kept as the historical record of what was first written:

    - The bound carries the SAME observation floor as the model it bounds. A frame that
      falls back to the global scale gets `inf`, i.e. no masking, exactly matching today's
      scale path. Bounding a scale-only frame by its own 3 observations supervised 3.9% of
      it while a zero-observation frame stayed fully supervised — less evidence, more
      masking, which is not defensible.
    - The bound is sourced from the fit's SURVIVING INLIERS, not the raw correspondences,
      so one track on a sky pixel cannot extend the supervised range into garbage.
    - The effective bound is `min(far_limit, saturation_horizon)` where the horizon is
      `-a/b` for `b < 0`. Without it, pixels AT or PAST the horizon but inside far_limit
      come out negative or infinite rather than masked. Measured on the shipped code at
      a=1.5, b=-0.03 (horizon 50), far_limit 60:
      `[10, 45, 49, 49.9, 50, 55, 80] -> [8.33, 300.0, 1633.3, 16633.3, 0, 0, 0]`.
      Note what the horizon term does NOT retire: pixels just BELOW the horizon still
      produce very large depth targets. Zeroing those would need a margin below the
      horizon, which is an unmeasured tunable and was deliberately not invented. The band
      is only reachable when `far_limit > horizon`, and the p99 positivity guard already
      forces `horizon > p99` — so it needs a track landing above the 99th percentile of
      the depth map, i.e. on sky. Bounded, documented, not fixed.
    """
    n_frames = depth.shape[0]
    coeffs = np.zeros((n_frames, 2), dtype=np.float64)
    far_limits = np.full(n_frames, np.inf)
    fitted = np.zeros(n_frames, dtype=bool)
    scale_only_rows: list[int] = []
    scales = np.full(n_frames, np.nan)

    pairs = _depth_correspondences(reconstruction, image_names, depth)
    for row, (d_colmap, d_vda) in enumerate(pairs):
        if len(d_colmap) == 0:
            continue

        # Evidence bound first: it applies whether or not the affine fit survives
        far_limits[row] = float(d_vda.max())
        if len(d_colmap) >= MIN_ALIGN_OBS:
            scales[row] = float(np.median(d_colmap / d_vda))

        # Affine needs its own, higher observation floor
        if len(d_colmap) < MIN_AFFINE_OBS:
            continue
        solution = _fit_affine_disparity(d_colmap, d_vda)
        if solution is None:
            continue
        a, b = solution

        # Reject a fit that inverts depth or saturates inside the frame's own range
        far_depth = float(np.percentile(depth[row][depth[row] > 0], 99)) if (depth[row] > 0).any() else 0.0
        if a <= 0 or far_depth <= 0 or (a / far_depth + b) <= AFFINE_EPS:
            continue
        coeffs[row] = (a, b)
        fitted[row] = True

    # Scale-only frames: a = 1/s, b = 0 reproduces the scale path exactly
    global_scale = float(np.median(scales[~np.isnan(scales)])) if (~np.isnan(scales)).any() else None
    for row in np.flatnonzero(~fitted):
        scale = scales[row] if not np.isnan(scales[row]) else global_scale
        if scale is None or scale <= 0:
            raise ValueError(
                "depth alignment: no frame has enough valid track observations to fit even a "
                "scale — the reconstruction is too sparse to align VDA depth to the COLMAP world."
            )
        coeffs[row] = (1.0 / scale, 0.0)
        scale_only_rows.append(int(row))

    stats = {
        "n_fitted": int(fitted.sum()),
        "n_fallback": len(scale_only_rows),
        "fallback_frames": [image_names[i] for i in scale_only_rows],
        "global_scale": global_scale,
        "far_limit_p10_p50_p90": [float(x) for x in np.percentile(far_limits[np.isfinite(far_limits)], [10, 50, 90])]
        if np.isfinite(far_limits).any()
        else [],
    }
    return coeffs, far_limits, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_align.py -v`
Expected: all pass

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py
git commit --only collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py -m "feat(sfm): affine-in-disparity depth alignment with one-sided far bound"
```

---

## Task 6: Select the alignment model in `apply_depth_alignment`

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py:435-475` (`apply_depth_alignment`)
- Test: `tests/pointcloud/test_depth_align.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/pointcloud/test_depth_align.py`:

```python
class _Result:
    """Minimal FeedforwardResult stand-in: the fields apply_depth_alignment touches."""

    def __init__(self, depth):
        from pathlib import Path

        self.depth = depth
        self.image_paths = [Path("frame_000000.jpg")]
        self.extrinsics = np.eye(4, dtype=np.float32)[None]
        self.intrinsics = np.array(
            [[[10.0, 0.0, GRID_W / 2], [0.0, 10.0, GRID_H / 2], [0.0, 0.0, 1.0]]], dtype=np.float32
        )
        self.world_points = None


def test_apply_depth_alignment_scale_is_the_default_and_stamps_the_model():
    observations = [(i % GRID_W, i % GRID_H, 2.0 * (i + 1)) for i in range(30)]
    recon = _fake_reconstruction({"frame_000000.jpg": observations})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(30):
        depth[0, i % GRID_H, i % GRID_W] = float(i + 1)
    result = _Result(depth.copy())

    attrs = sfm.apply_depth_alignment(result, recon)
    assert attrs["depth_scale"] == "colmap"
    assert attrs["depth_align_model"] == "scale"
    assert result.depth[0, 4, 4] == pytest.approx(2.0 * depth[0, 4, 4])


def test_apply_depth_alignment_affine_stamps_coefficients():
    recon, depth = _affine_scene(a=1.5, b=-0.004)
    result = _Result(depth.copy())

    attrs = sfm.apply_depth_alignment(result, recon, model="affine")
    assert attrs["depth_scale"] == "colmap"
    assert attrs["depth_align_model"] == "affine"
    assert attrs["depth_affine_ab"][0][0] == pytest.approx(1.5, rel=1e-4)
    assert result.world_points is not None


def test_apply_depth_alignment_rejects_an_unknown_model():
    recon, depth = _affine_scene(a=1.0, b=0.0)
    with pytest.raises(ValueError, match="depth_align"):
        sfm.apply_depth_alignment(_Result(depth.copy()), recon, model="quadratic")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_align.py -k apply_depth_alignment -v`
Expected: FAIL with `KeyError: 'depth_align_model'`

- [ ] **Step 3: Write the implementation**

Replace `apply_depth_alignment` in `collab_splats/pointcloud/sfm.py` with:

```python
def apply_depth_alignment(
    result: "FeedforwardResult",
    reconstruction: pycolmap.Reconstruction,
    model: str = "scale",
) -> dict:
    """
    Align result.depth to the reconstruction's world scale in place; recompute world_points.

    - model="scale": one robust multiplier per frame (the shipped behaviour).
    - model="affine": a per-frame affine fit in disparity, applied as d/(a + b*d), with
      saturated and beyond-evidence pixels written as 0 (= no depth target, no mesh sample).
    - Returns the provenance attrs to merge into save_zarr's extra_attrs; raises on an
      unalignable scene — never a silent VDA-metric write.
    """
    if model not in ("scale", "affine"):
        raise ValueError(f"pointcloud.instantsfm.depth_align must be 'scale' or 'affine', got {model!r}")

    # SfM image_paths are extension-less stems (Path(im.name) from COLMAP, whose image
    # names ARE stems) — path.name is the COLMAP image name, the splats-branch convention
    names = [path.name for path in result.image_paths]

    if model == "scale":
        scales, stats = align_depth_to_reconstruction(reconstruction, names, result.depth)
        logger.info(
            "depth alignment (scale): global scale %.4f, ratio p10/p50/p90 %s -> %s, %d fallback frames",
            stats["global_scale"],
            [round(x, 4) for x in stats["ratio_p10_p50_p90_before"]],
            [round(x, 4) for x in stats["ratio_p10_p50_p90_after"]],
            stats["n_fallback"],
        )
        result.depth = (result.depth * scales[:, None, None]).astype(np.float32)
        attrs = {
            "depth_scale": "colmap",
            "depth_align_model": "scale",
            "depth_scales": [float(s) for s in scales],
            "depth_scale_fallback_frames": stats["fallback_frames"],
        }
    else:
        coeffs, far_limits, stats = align_depth_affine(reconstruction, names, result.depth)
        aligned = np.stack(
            [
                _apply_affine_depth(result.depth[row], coeffs[row, 0], coeffs[row, 1], far_limits[row])
                for row in range(len(names))
            ]
        )

        # Masked fraction is the honest reliability signal on this path — the sfm branch has
        # no confidence channel, so this is what mesh/splats consumers actually see
        had_depth = result.depth > 0
        masked = float((had_depth & (aligned <= 0)).sum()) / max(float(had_depth.sum()), 1.0)
        logger.info(
            "depth alignment (affine): %d/%d frames fitted, %d scale-only, far-limit p10/p50/p90 %s m, "
            "%.2f%% of positive pixels masked (saturation + beyond-evidence)",
            stats["n_fitted"], len(names), stats["n_fallback"],
            [round(x, 1) for x in stats["far_limit_p10_p50_p90"]],
            100.0 * masked,
        )
        result.depth = aligned.astype(np.float32)
        attrs = {
            "depth_scale": "colmap",
            "depth_align_model": "affine",
            "depth_affine_ab": [[float(a), float(b)] for a, b in coeffs],
            # inf is not valid JSON and zarr attrs are JSON. zarr 3.1.6 does NOT raise
            # here — it writes a bare `Infinity` token, which strict JSON readers
            # reject. A scale-only frame's bound must be written as null.
            "depth_far_limits": [None if not np.isfinite(x) else float(x) for x in far_limits],
            "depth_masked_fraction": masked,
            "depth_scale_fallback_frames": stats["fallback_frames"],
        }

    # Re-unproject dense world points (t is not scale-invariant, so world_points cannot be
    # scaled directly — they must be re-derived from the aligned depth under the COLMAP poses)
    result.world_points = unproject_depth_map_to_point_map(
        result.depth[..., None], result.extrinsics[:, :3, :], result.intrinsics
    ).astype(np.float32)

    return attrs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/ -v`
Expected: all pass

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py
git commit --only collab_splats/pointcloud/sfm.py tests/pointcloud/test_depth_align.py -m "feat(sfm): depth_align selects scale or affine alignment"
```

---

## Task 7: Reproducible InstantSfM (`random_seed`)

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py:808-829` (`InstantSfMCreator` fields + `_build_config`)
- Test: `tests/pointcloud/test_sfm_creator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/pointcloud/test_sfm_creator.py`:

```python
def test_random_seed_reaches_runtime_options(monkeypatch):
    from collab_splats.pointcloud.sfm import InstantSfMCreator

    class _Config:
        def __init__(self, features):
            self.features = features
            self.OPTIONS = {"skip_retriangulation": True}
            self.RUNTIME_OPTIONS = {"use_depths": False}

    import instantsfm.controllers.config as upstream_config

    monkeypatch.setattr(upstream_config, "Config", _Config)

    seeded = InstantSfMCreator(random_seed=1234)._build_config()
    assert seeded.RUNTIME_OPTIONS["random_seed"] == 1234

    unseeded = InstantSfMCreator()._build_config()
    assert "random_seed" not in unseeded.RUNTIME_OPTIONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_sfm_creator.py -k random_seed -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'random_seed'`

- [ ] **Step 3: Write the implementation**

In `collab_splats/pointcloud/sfm.py`, add a field to `InstantSfMCreator`:

```python
    features: str = "colmap"
    single_camera: bool = True
    use_depths: bool = True
    retriangulation: bool = False
    random_seed: int | None = None
```

And in `_build_config`, after the `skip_retriangulation` line:

```python
        # InitializeRandomPositions draws camera translations and track xyzs from an unseeded
        # np.random.uniform(-1, 1) (global_positioning.py:232-243), so two runs of the same
        # scene differ. random_seed is an upstream RUNTIME_OPTION (global_mapper.py:25) that
        # seeds numpy/random/torch/cuda; neither we nor upstream's CLI sets it by default.
        if self.random_seed is not None:
            config.RUNTIME_OPTIONS["random_seed"] = int(self.random_seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_sfm_creator.py -v`
Expected: all pass

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/pointcloud/sfm.py tests/pointcloud/test_sfm_creator.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/pointcloud/sfm.py tests/pointcloud/test_sfm_creator.py
git commit --only collab_splats/pointcloud/sfm.py tests/pointcloud/test_sfm_creator.py -m "feat(sfm): expose InstantSfM random_seed for reproducible reconstructions"
```

---

## Task 8: 2DGS median depth: render dict and `splats.zarr`

Two commits: median depth reaches the render dict, then it is persisted. Part B needs only Part A, nothing from the loss tasks.

### Part A: 2DGS median depth reaches the render dict

**Files:**
- Modify: `collab_splats/splats/rendering.py:90-111`
- Test: `tests/splats/test_rendering.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_rendering.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization is CUDA-only")
def test_2dgs_render_carries_median_depth_and_its_normal():
    gaussians, cam_to_world, intrinsics = _toy_scene()
    render, _info = render_view(
        "2dgs", gaussians, cam_to_world, intrinsics, width=32, height=32, sh_degree=0, absgrad=False
    )
    assert render["median_depth"].shape == render["depth"].shape
    assert render["depth_normal_median"].shape == render["depth_normal"].shape
    assert render["median_depth"].requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization is CUDA-only")
def test_3dgs_render_has_no_median_depth():
    gaussians, cam_to_world, intrinsics = _toy_scene()
    render, _info = render_view(
        "3dgs", gaussians, cam_to_world, intrinsics, width=32, height=32, sh_degree=0, absgrad=False
    )
    assert "median_depth" not in render
```

If `tests/splats/test_rendering.py` has no `_toy_scene` helper, add one built on the existing
synthetic scene:

```python
def _toy_scene(n_points=200, device="cuda"):
    """A small ParameterDict + one camera, enough to exercise both rasterizers."""
    from collab_splats.splats.trainer import init_gaussians_from_points
    from tests.splats.synthetic import make_scene

    _images, world_to_cam, intrinsics, points, colors, _depths = make_scene(
        n_views=1, height=32, width=32, n_points=n_points
    )
    gaussians = init_gaussians_from_points(points, colors, sh_degree=0, init_opacity=0.1, device=device)
    cam_to_world = torch.linalg.inv(torch.from_numpy(world_to_cam[:1]).float().to(device))
    return gaussians, cam_to_world, torch.from_numpy(intrinsics[:1]).float().to(device)
```

Check `init_gaussians_from_points`'s exact signature before wiring the helper
(`collab_splats/splats/trainer.py:222`) and match it.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_rendering.py -k median -v`
Expected: FAIL with `KeyError: 'median_depth'` (or SKIPPED with no CUDA — then verify on a GPU box before merging)

- [ ] **Step 3: Write the implementation**

In `collab_splats/splats/rendering.py`, replace the 2dgs branch:

```python
    if primitive == "2dgs":
        rgb_depth, alpha, normal_world, _depth_normal_world, distortion, median_depth, info = rasterization_2dgs(
            **shared_kwargs, distloss=True
        )
        rgb = rgb_depth[..., :3]
        depth = rgb_depth[..., 3:4]
        rotation_w2c = world_to_cam[:, :3, :3]
        normal_cam = torch.einsum("cij,chwj->chwi", rotation_w2c, normal_world)

        # RaDe-GS median depth: the depth of the median Gaussian along each ray, rather than
        # the alpha-weighted expectation. Sparse by construction (one Gaussian per ray
        # receives gradient) but sharper across depth discontinuities, so its finite-differenced
        # normal is a second consistency target — see losses.normal_consistency_loss.
        render = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": depth,
            "median_depth": median_depth,
            "normal": normal_cam,
            "depth_normal": depth_to_normal(depth, identity_pose, intrinsics),
            "depth_normal_median": depth_to_normal(median_depth, identity_pose, intrinsics),
            "distortion": distortion,
        }
        return render, info
```

Extend the module docstring's 2DGS sentence to mention the median depth:

```
2DGS goes through ``gsplat.rasterization_2dgs`` which returns rendered normals (world frame), a
distortion map, and the median depth (RaDe-GS's surface depth). Normals are rotated into camera
space and depth normals are finite-differenced at an identity pose for both primitives and for
both depths, so the consistency loss compares like with like.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_rendering.py -v`
Expected: all pass (or skipped without CUDA)

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/splats/rendering.py tests/splats/test_rendering.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/splats/rendering.py tests/splats/test_rendering.py
git commit --only collab_splats/splats/rendering.py tests/splats/test_rendering.py -m "feat(splats): keep 2dgs median depth and its finite-differenced normal"
```

---

### Part B: Write `median_depth` into `splats.zarr`

**Files:**
- Modify: `collab_splats/splats/outputs.py:30-110`
- Test: `tests/splats/test_outputs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_outputs.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization is CUDA-only")
def test_2dgs_render_all_views_writes_median_depth(tmp_path):
    store = _render_scene(tmp_path, primitive="2dgs")
    assert "median_depth" in store
    assert store["median_depth"].shape == store["depth"].shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization is CUDA-only")
def test_3dgs_render_all_views_omits_median_depth(tmp_path):
    store = _render_scene(tmp_path, primitive="3dgs")
    assert "median_depth" not in store
```

Add a `_render_scene(tmp_path, primitive)` helper to that file if one does not already exist,
built on `tests/splats/synthetic.make_scene` and `render_all_views`, matching the way the
existing tests in `tests/splats/test_outputs.py` construct their inputs.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_outputs.py -k median_depth -v`
Expected: FAIL with `assert 'median_depth' in store` (or SKIPPED without CUDA — verify on a GPU box before merging)

- [ ] **Step 3: Write the implementation**

In `collab_splats/splats/outputs.py`, in `render_all_views`, extend the array declaration:

```python
    # Per-view chunks so downstream stages read frames independently; filled inside the loop
    per_view_arrays = {
        "rgb": ((n_views, height, width, 3), np.uint8),
        "depth": ((n_views, height, width), np.float32),
        "normal": ((n_views, height, width, 3), np.float32),
        "alpha": ((n_views, height, width), np.float32),
    }

    # 2DGS also renders a median (surface) depth; mesh.splat_depth chooses which one TSDF fuses
    writes_median_depth = cfg.primitive == "2dgs"
    if writes_median_depth:
        per_view_arrays["median_depth"] = ((n_views, height, width), np.float32)
```

And in the per-view write block, after the `store["alpha"][view] = ...` line:

```python
            if writes_median_depth:
                store["median_depth"][view] = render["median_depth"][0, ..., 0].cpu().numpy()
```

Add one docstring bullet to `render_all_views`:

```
    - 2DGS additionally writes ``median_depth`` (the RaDe-GS surface depth); 3DGS has none.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/ -v`
Expected: all pass (or skipped without CUDA)

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/splats/outputs.py tests/splats/test_outputs.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/splats/outputs.py tests/splats/test_outputs.py
git commit --only collab_splats/splats/outputs.py tests/splats/test_outputs.py -m "feat(splats): write 2dgs median_depth into splats.zarr"
```

---

## Task 9: Blended normal consistency (`depth_ratio`)

Two commits: the blended loss, then the config validation that guards its one knob.

### Part A: Blended normal consistency (`depth_ratio`)

**Files:**
- Modify: `collab_splats/splats/losses.py` (all six loss functions + `compute_losses`)
- Test: `tests/splats/test_losses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_losses.py`:

```python
def test_depth_ratio_zero_matches_the_expected_only_loss():
    render = _render()
    render["depth_normal_median"] = torch.nn.functional.normalize(
        torch.randn(1, 16, 16, 3, generator=torch.Generator().manual_seed(2)), dim=-1
    )
    target, gaussians = _target(), _gaussians()
    fn = OPTIONAL_LOSSES["normal_consistency"]

    plain = fn(render, target, gaussians, 1.0, {"weight": 1.0})
    ratio_zero = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 0.0})
    assert torch.allclose(plain, ratio_zero)


def test_depth_ratio_one_uses_only_the_median_normal():
    render = _render()
    median = torch.nn.functional.normalize(
        torch.randn(1, 16, 16, 3, generator=torch.Generator().manual_seed(2)), dim=-1
    )
    render["depth_normal_median"] = median
    target, gaussians = _target(), _gaussians()
    fn = OPTIONAL_LOSSES["normal_consistency"]

    blended = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 1.0})

    # Same computation with the median normal in the expected slot
    swapped = dict(render)
    swapped["depth_normal"] = median
    reference = fn(swapped, target, gaussians, 1.0, {"weight": 1.0})
    assert torch.allclose(blended, reference)


def test_depth_ratio_is_a_convex_blend():
    render = _render()
    render["depth_normal_median"] = torch.nn.functional.normalize(
        torch.randn(1, 16, 16, 3, generator=torch.Generator().manual_seed(2)), dim=-1
    )
    target, gaussians = _target(), _gaussians()
    fn = OPTIONAL_LOSSES["normal_consistency"]

    at_zero = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 0.0})
    at_one = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 1.0})
    at_six = fn(render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 0.6})
    assert torch.allclose(at_six, 0.4 * at_zero + 0.6 * at_one, atol=1e-6)


def test_depth_ratio_without_a_median_normal_raises():
    render = _render()  # no depth_normal_median
    target, gaussians = _target(), _gaussians()
    with pytest.raises(ValueError, match="depth_normal_median"):
        OPTIONAL_LOSSES["normal_consistency"](
            render, target, gaussians, 1.0, {"weight": 1.0, "depth_ratio": 0.6}
        )


def test_compute_losses_passes_the_spec_through():
    render = _render()
    render["depth_normal_median"] = torch.nn.functional.normalize(
        torch.randn(1, 16, 16, 3, generator=torch.Generator().manual_seed(2)), dim=-1
    )
    schedule = {"normal_consistency": {"weight": 1.0, "depth_ratio": 1.0}}
    total, values = compute_losses(0, render, _target(), _gaussians(), schedule, 1.0)
    assert "normal_consistency" in values
```

Every existing call in `tests/splats/test_losses.py` that invokes a loss function directly must
gain the new 5th argument `{"weight": 1.0}`. Update them in this step.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_losses.py -k depth_ratio -v`
Expected: FAIL with `TypeError: normal_consistency_loss() takes 4 positional arguments but 5 were given`

- [ ] **Step 3: Write the implementation**

In `collab_splats/splats/losses.py`, add `spec: dict` as the 5th parameter of all six loss
functions (`depth_loss`, `normal_consistency_loss`, `distortion_loss`, `opacity_reg_loss`,
`scale_reg_loss`, `appearance_reg_loss`). Only `normal_consistency_loss` reads it.

Replace `normal_consistency_loss` with:

```python
def normal_consistency_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor | None:
    """
    Cosine distance between rendered normals and normals finite-differenced from rendered depth.

    - `spec["depth_ratio"]` (default 0.0) blends in the RaDe-GS median-depth normal:
      `(1 - r) * cos(n, dn_expected) + r * cos(n, dn_median)`. This is a blend of two LOSSES,
      the RaDe-GS semantics — not upstream-2DGS's `depth_ratio`, which blends the two depths
      into one surf_depth before differencing.
    - Raises when the render carries no normals: the trainer gates `render_normals` on `loss_active`, so an
      active loss without normals is a wiring bug, not a condition to skip silently.
    """
    rendered_normal = render.get("normal")
    depth_normal = render.get("depth_normal")
    if rendered_normal is None or depth_normal is None:
        raise ValueError("normal_consistency is active but the render has no normals; render with render_normals=True")

    # Scaling depth_normal by detached alpha scales the GRADIENT so empty pixels stop pulling; the
    # reported value still carries a (1 - alpha) offset on those pixels. Parity with upstream
    # simple_trainer_2dgs.py. The scaled vector is no longer unit-norm, so GSPLAT_ENFORCE_CONTRACTS=1
    # trips normal_cosine_loss's norm assert here by design.
    alpha = render["alpha"].detach()
    expected_term = gsplat_losses.normal_cosine_loss(rendered_normal, depth_normal * alpha).mean()

    # depth_ratio 0 is the shipped behaviour, bit-for-bit
    ratio = float(spec.get("depth_ratio", 0.0))
    if ratio <= 0.0:
        return expected_term

    # Median depth is a 2DGS rasterizer output; an active ratio without it is a wiring bug
    median_normal = render.get("depth_normal_median")
    if median_normal is None:
        raise ValueError(
            "normal_consistency depth_ratio > 0 needs 'depth_normal_median' in the render "
            "(2dgs only — median depth is a rasterization_2dgs output)"
        )
    median_term = gsplat_losses.normal_cosine_loss(rendered_normal, median_normal * alpha).mean()
    return (1.0 - ratio) * expected_term + ratio * median_term
```

In `compute_losses`, pass the spec:

```python
        loss_fn = OPTIONAL_LOSSES[name]
        value = loss_fn(render, target, gaussians, scene_scale, spec)
```

Update the module docstring's second sentence:

```
Photometric (0.8 L1 + 0.2 (1 - SSIM)) is always on. Each optional loss is one small function
with the same signature ``(render, target, gaussians, scene_scale, spec)``; ``compute_losses``
loops over the yaml schedule ``name: {weight[, start, end, end_weight]}`` and adds a loss iff its
weight at the step is > 0 and the function returns a value.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_losses.py -v`
Expected: all pass

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/splats/losses.py tests/splats/test_losses.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/splats/losses.py tests/splats/test_losses.py
git commit --only collab_splats/splats/losses.py tests/splats/test_losses.py -m "feat(splats): RaDe-GS median-normal blend via losses.normal_consistency.depth_ratio"
```

---

### Part B: Validate `depth_ratio` in `SplatsConfig`

**Files:**
- Modify: `collab_splats/splats/trainer.py:140-165`
- Test: `tests/splats/test_trainer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_trainer.py`:

```python
def test_depth_ratio_accepted_on_normal_consistency_for_2dgs():
    cfg = SplatsConfig.from_dict(
        {"primitive": "2dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": 0.6}}}
    )
    assert cfg.losses["normal_consistency"]["depth_ratio"] == 0.6


def test_depth_ratio_rejected_on_another_loss():
    with pytest.raises(ValueError, match="depth_ratio"):
        SplatsConfig.from_dict({"primitive": "2dgs", "losses": {"depth": {"weight": 0.01, "depth_ratio": 0.6}}})


def test_depth_ratio_out_of_range_rejected():
    with pytest.raises(ValueError, match="depth_ratio"):
        SplatsConfig.from_dict(
            {"primitive": "2dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": 1.5}}}
        )


def test_depth_ratio_is_2dgs_only():
    with pytest.raises(ValueError, match="2dgs"):
        SplatsConfig.from_dict(
            {"primitive": "3dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": 0.6}}}
        )


def test_depth_ratio_zero_is_allowed_on_3dgs():
    cfg = SplatsConfig.from_dict(
        {"primitive": "3dgs", "losses": {"normal_consistency": {"weight": 0.05, "depth_ratio": 0.0}}}
    )
    assert cfg.losses["normal_consistency"]["depth_ratio"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_trainer.py -k depth_ratio -v`
Expected: FAIL — `SplatsConfig.from_dict` raises `splats.losses.normal_consistency: expected {weight[, start, end, end_weight]}`

- [ ] **Step 3: Write the implementation**

In `collab_splats/splats/trainer.py`, inside `from_dict`'s loss loop, replace the
`unknown_spec_keys` line with:

```python
            # depth_ratio is the RaDe-GS median-normal blend and belongs to one loss only
            allowed_spec_keys = {"weight", "start", "end", "end_weight"}
            if name == "normal_consistency":
                allowed_spec_keys = allowed_spec_keys | {"depth_ratio"}
            unknown_spec_keys = set(spec) - allowed_spec_keys
```

And after the loop, beside the distortion guard:

```python
        # Median depth only exists for 2DGS, so a non-zero blend on 3dgs is a config error
        depth_ratio = float(cfg.losses.get("normal_consistency", {}).get("depth_ratio", 0.0))
        if not 0.0 <= depth_ratio <= 1.0:
            raise ValueError(f"splats.losses.normal_consistency.depth_ratio must be in [0, 1], got {depth_ratio}")
        if depth_ratio > 0 and cfg.primitive != "2dgs":
            raise ValueError(
                "splats.losses.normal_consistency.depth_ratio > 0 is 2dgs-only "
                "(median depth is a rasterization_2dgs output); set it to 0 or use primitive: 2dgs"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_trainer.py -v`
Expected: all pass

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/splats/trainer.py tests/splats/test_trainer.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/splats/trainer.py tests/splats/test_trainer.py
git commit --only collab_splats/splats/trainer.py tests/splats/test_trainer.py -m "feat(splats): validate normal_consistency.depth_ratio (range, 2dgs-only)"
```

---

## Task 10: `mesh.splat_depth` selects the fused depth

**Files:**
- Modify: `collab_splats/mesh/utils.py:622-659` (`_splats_to_tsdf_inputs`), `collab_splats/wrapper/reconstructor.py:521-560` and `:1339-1412`
- Test: `tests/mesh/test_splats_adapter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/mesh/test_splats_adapter.py`:

```python
def test_splat_depth_median_reads_the_median_array(tmp_path):
    import numpy as np
    import zarr

    from collab_splats.mesh.utils import _splats_to_tsdf_inputs

    path = tmp_path / "splats.zarr"
    store = zarr.open_group(str(path), mode="w")
    n, h, w = 2, 4, 4
    store.create_array("depth", data=np.full((n, h, w), 1.0, dtype=np.float32))
    store.create_array("median_depth", data=np.full((n, h, w), 3.0, dtype=np.float32))
    store.create_array("alpha", data=np.ones((n, h, w), dtype=np.float32))
    store.create_array("rgb", data=np.zeros((n, h, w, 3), dtype=np.uint8))
    store.create_array("c2w", data=np.stack([np.eye(4, dtype=np.float32)] * n))
    store.create_array("K", data=np.stack([np.eye(3, dtype=np.float32)] * n))

    expected, _rgbs, _c2w, _K = _splats_to_tsdf_inputs(path, splat_depth="expected")
    median, _rgbs, _c2w, _K = _splats_to_tsdf_inputs(path, splat_depth="median")
    assert expected[0, 0, 0] == 1.0
    assert median[0, 0, 0] == 3.0


def test_splat_depth_median_missing_raises_actionably(tmp_path):
    import numpy as np
    import zarr

    from collab_splats.mesh.utils import _splats_to_tsdf_inputs

    path = tmp_path / "splats.zarr"
    store = zarr.open_group(str(path), mode="w")
    n, h, w = 1, 4, 4
    store.create_array("depth", data=np.ones((n, h, w), dtype=np.float32))
    store.create_array("alpha", data=np.ones((n, h, w), dtype=np.float32))
    store.create_array("rgb", data=np.zeros((n, h, w, 3), dtype=np.uint8))
    store.create_array("c2w", data=np.stack([np.eye(4, dtype=np.float32)]))
    store.create_array("K", data=np.stack([np.eye(3, dtype=np.float32)]))

    with pytest.raises(ValueError, match="median_depth"):
        _splats_to_tsdf_inputs(path, splat_depth="median")


def test_splat_depth_rejects_an_unknown_value(tmp_path):
    from collab_splats.mesh.utils import _splats_to_tsdf_inputs

    with pytest.raises(ValueError, match="splat_depth"):
        _splats_to_tsdf_inputs(tmp_path / "splats.zarr", splat_depth="surf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_splats_adapter.py -k splat_depth -v`
Expected: FAIL with `TypeError: _splats_to_tsdf_inputs() got an unexpected keyword argument 'splat_depth'`

- [ ] **Step 3: Write the implementation**

In `collab_splats/mesh/utils.py`, change `_splats_to_tsdf_inputs`:

```python
def _splats_to_tsdf_inputs(
    splats_zarr: Path, conf_percentile: float | None = None, splat_depth: str = "expected"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

Add a docstring bullet:

```
    - `splat_depth` picks which rendered depth to fuse: "expected" (alpha-weighted, the
      default) or "median" (RaDe-GS surface depth, 2dgs renders only).
```

Replace the top of the body:

```python
    # Value check before any IO so a typo fails on the config, not on a missing array
    depth_arrays = {"expected": "depth", "median": "median_depth"}
    if splat_depth not in depth_arrays:
        raise ValueError(f"mesh.splat_depth must be 'expected' or 'median', got {splat_depth!r}")

    # Loud failure before opening: the splats stage is never auto-run by mesh()
    splats_zarr = Path(splats_zarr)
    if not splats_zarr.exists():
        raise FileNotFoundError(f"{splats_zarr} — run the splats stage first")
    store = zarr.open_group(str(splats_zarr), mode="r")

    depth_array = depth_arrays[splat_depth]
    if depth_array not in store:
        raise ValueError(
            f"{splats_zarr} has no '{depth_array}' array — mesh.splat_depth: median needs a 2dgs "
            "splats run from this version; re-run the splats stage or use splat_depth: expected."
        )
```

And change the depth read:

```python
    depths = np.ascontiguousarray(store[depth_array][:], dtype=np.float32)
```

Extend the existing alpha-mask log so the source is visible:

```python
    logger.info(
        "Splat depth source '%s' (%s); alpha mask (p%s): %.1f%% of depth pixels dropped",
        splat_depth,
        depth_array,
        "none" if conf_percentile is None else f"{conf_percentile:.0f}",
        100.0 * float(dropped.mean()),
    )
```

In `collab_splats/wrapper/reconstructor.py`, add `splat_depth: str = "expected"` as the last
keyword parameter of `_run_tsdf_mesh` and pass it through:

```python
        depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(
            splats_zarr, conf_percentile=conf_percentile, splat_depth=splat_depth
        )
```

And in the `mesh()` stage's `_run_tsdf_mesh(...)` call, add:

```python
            splat_depth=mesh_cfg["splat_depth"],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/ -v`
Expected: all pass

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 collab_splats/mesh/utils.py collab_splats/wrapper/reconstructor.py tests/mesh/test_splats_adapter.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/mesh/utils.py collab_splats/wrapper/reconstructor.py tests/mesh/test_splats_adapter.py
git commit --only collab_splats/mesh/utils.py collab_splats/wrapper/reconstructor.py tests/mesh/test_splats_adapter.py -m "feat(mesh): splat_depth selects expected or median rendered depth"
```

---

## Task 11: Wire the context stream and alignment model through the Reconstructor

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:129-240` (`extract_frames`), `:832-843` (call site), `:1020-1090` (`_run_sfm`)
- Test: `tests/wrapper/test_vda_context.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/wrapper/test_vda_context.py`:

```python
"""
VDA context stream wiring: keyframes are drawn from the context grid and map back to its rows.
"""

import numpy as np
import pytest

from collab_splats.preproc.video import context_indices
from collab_splats.wrapper.reconstructor import _context_keep_rows


def test_keep_rows_map_keyframes_onto_the_grid():
    grid = list(range(0, 60, 3))
    keyframes = [0, 9, 30, 57]
    assert _context_keep_rows(grid, keyframes) == [0, 3, 10, 19]


def test_keep_rows_returns_none_when_a_keyframe_is_off_grid(caplog):
    grid = list(range(0, 60, 3))
    keyframes = [0, 10, 30]  # 10 is not a multiple of 3
    with caplog.at_level("WARNING"):
        assert _context_keep_rows(grid, keyframes) is None
    assert "off the context grid" in caplog.text


def test_keep_rows_returns_none_for_an_empty_grid():
    assert _context_keep_rows([], [0, 3]) is None


def test_context_grid_contains_a_keyframe_grid_at_a_multiple_rate(tiny_video):
    # 8 FPS context and 2 FPS keyframes on the same video: keyframes are a strict subset
    context = context_indices(tiny_video, target_fps=10.0)
    keyframes = context_indices(tiny_video, target_fps=5.0)
    assert set(keyframes) <= set(context)
```

Add the `tiny_video` fixture to `tests/wrapper/` by importing it, or copy the fixture from
`tests/preproc/conftest.py` into `tests/wrapper/conftest.py` if the wrapper conftest has none.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_vda_context.py -v`
Expected: FAIL with `ImportError: cannot import name '_context_keep_rows'`

- [ ] **Step 3: Write the implementation**

In `collab_splats/wrapper/reconstructor.py`, add a helper beside `_apply_undistortion`:

```python
def _context_keep_rows(grid: Sequence[int], keyframe_indices: Sequence[int]) -> list[int] | None:
    """
    Positions of each keyframe within the context grid, or None when they do not line up.

    - Keyframes selected through preproc's candidate grid are grid members by construction;
      a scene whose frames.zarr predates that change is not, and gets the keyframe-only VDA
      path rather than a silently misaligned depth stack.
    """
    grid_array = np.asarray(grid, dtype=np.int64)
    if grid_array.size == 0:
        logger.warning("VDA context grid is empty — falling back to keyframe-only VDA")
        return None

    positions = np.searchsorted(grid_array, np.asarray(keyframe_indices, dtype=np.int64))
    positions = np.clip(positions, 0, grid_array.size - 1)
    off_grid = grid_array[positions] != np.asarray(keyframe_indices, dtype=np.int64)
    if off_grid.any():
        logger.warning(
            "%d of %d keyframes are off the context grid (first: %d) — falling back to "
            "keyframe-only VDA. Re-run preprocess with preproc.vda_context_fps set to align them.",
            int(off_grid.sum()), len(keyframe_indices), int(np.asarray(keyframe_indices)[off_grid][0]),
        )
        return None
    return [int(p) for p in positions]
```

Add `vda_context_fps: float | None = None` as the last keyword parameter of `extract_frames`,
extend its docstring, and compute the grid before the selection branch:

```python
    # Context grid: when the VDA context stream is enabled, keyframes must be grid members so
    # the depth rows map back to them by position (see _context_keep_rows)
    candidates = None
    if vda_context_fps:
        if frame_selection == "optical_flow":
            raise ValueError(
                "preproc.vda_context_fps requires frame_selection 'fps' or 'uniform' — "
                "optical_flow picks frames by motion and cannot be restricted to a grid."
            )
        candidates = context_indices(str(input_path), target_fps=vda_context_fps)
        logger.info(
            "VDA context grid: %d frames at %.2f fps; keyframes will be drawn from it",
            len(candidates), vda_context_fps,
        )
```

Pass `candidates=candidates` in both the `sample_fps(...)` and `sample_uniform(...)` calls, and
record it in provenance:

```python
    prov = {
        "video_path": str(input_path),
        "video_mtime": input_path.stat().st_mtime,
        "method": method,
        "fps": fps,
        "max_frames": max_frames,
        "vda_context_fps": vda_context_fps,
    }
```

Add the import at the top of `reconstructor.py`:

```python
from collab_splats.preproc.video import context_indices, decode_context, get_video_info
```

(keep whatever `video` imports already exist and merge).

At the `extract_frames(...)` call site in `preprocess`, add:

```python
            vda_context_fps=pre_cfg["vda_context_fps"],
```

In `_run_sfm`, replace the VDA block with:

```python
        # VDA metric depth — the only shipped mode (use_depths=True). Gate on the npy set BEFORE
        # decoding anything (300 x 1080p is ~1.9 GB).
        context_fps = self.config["preproc"]["vda_context_fps"]

        # vda_depth_complete keys on `names` alone, but depth CONTENT now also depends on the
        # context grid. `names` is always the sequential frame_000000..N, so switching
        # vda_context_fps leaves the stem set identical and would silently reuse stale depth
        # (reproduced twice in review 2026-08-26). A sidecar records the generating inputs.
        depth_sidecar = backend_dir / "depth_vda" / "inputs.json"
        signature = {
            "context_fps": float(context_fps) if context_fps else None,
            "keyframe_fps": float(self.config["preproc"]["fps"]),
            "n_names": len(names),
        }
        cached_signature = json.loads(depth_sidecar.read_text()) if depth_sidecar.exists() else None

        # A missing sidecar means depths predate this stamp — trust them rather than forcing a
        # re-run of every existing scene. Only a PRESENT and DIFFERENT stamp invalidates.
        stale = cached_signature is not None and cached_signature != signature
        if stale:
            logger.info(
                "VDA depth cache invalidated: generating inputs changed %s -> %s",
                cached_signature, signature,
            )

        if stale or not vda_depth_complete(backend_dir, names):
            keep_rows, context_frames = None, None

            # Context stream: VDA is temporal, so run it over a contiguous constant-rate grid
            # and keep only the keyframe rows. Falls back to the keyframe path whenever the
            # source video is gone (rerun-from-processed) or the keyframes are off-grid.
            if context_fps:
                provenance = dict(store.store.attrs.get("provenance", {}))
                video_path = provenance.get("video_path")
                if video_path and Path(video_path).exists():
                    grid = context_indices(video_path, target_fps=float(context_fps))
                    keep_rows = _context_keep_rows(grid, [int(fi) for fi in store.frame_indices()])
                    if keep_rows is not None:
                        # Same distortion profile frames.zarr was written with, or K_new drifts
                        profile = None
                        if provenance.get("undistort"):
                            profile = DistortionProfile.from_dict(provenance["undistort"]["profile"])
                        logger.info(
                            "VDA context stream: decoding %d frames at %.2f fps from %s",
                            len(grid), float(context_fps), video_path,
                        )
                        context_frames = decode_context(video_path, grid, profile=profile)
                else:
                    logger.warning(
                        "preproc.vda_context_fps is set but the source video is unavailable "
                        "(%s) — falling back to keyframe-only VDA",
                        video_path,
                    )

            if context_frames is not None:
                generate_vda_depth(
                    context_frames,
                    fps=float(context_fps),
                    out_dir=backend_dir,
                    names=names,
                    keep_rows=keep_rows,
                )
                del context_frames
            else:
                frames = np.ascontiguousarray(store.images())
                generate_vda_depth(
                    frames, fps=float(self.config["preproc"]["fps"]), out_dir=backend_dir, names=names
                )
                del frames
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Stamp what produced these maps, so a later context-rate change invalidates them
            depth_sidecar.write_text(json.dumps(signature))
```

Add `json` to the imports at the top of `reconstructor.py` if it is not already there.

**Why a sidecar and not a wider `vda_depth_complete` signature:** the gate helper lives in
`sfm.py` and is also called by `evals/scripts/eval.py`, which has no notion of a context grid.
Keeping the signature in the Reconstructor — the only place that knows the context rate — avoids
changing a shared helper's contract for one caller's benefit.

Add `random_seed` to the creator construction in `_run_sfm`:

```python
        creator = InstantSfMCreator(
            features=pc_cfg["instantsfm"]["features"],
            retriangulation=pc_cfg["instantsfm"]["retriangulation"],
            random_seed=pc_cfg["instantsfm"]["random_seed"],
        )
```

And pass the alignment model at `reconstructor.py:1084`:

```python
        align_attrs = apply_depth_alignment(outputs, recon, model=pc_cfg["instantsfm"]["depth_align"])
```

Finally, replace the silent `conf_percentile` no-op in the splats stage:

```python
            elif conf_percentile is not None:
                # SfM depth carries no confidence channel, so mesh.conf_percentile cannot apply
                # here. Reliability is enforced upstream instead: affine alignment writes 0 for
                # saturated and beyond-evidence pixels, and 0 means "no target".
                logger.info(
                    "splats depth targets: mesh.conf_percentile=%s not applied on the sfm path "
                    "(no confidence channel); masking comes from depth alignment — %.2f%% of "
                    "target pixels are zero",
                    conf_percentile,
                    100.0 * float((depth_targets <= 0).mean()),
                )
```

Add `DistortionProfile` to the `collab_splats.preproc.undistort` import at the top of
`reconstructor.py`, and `Sequence` to its typing imports.

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_vda_context.py -v
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ tests/preproc/ tests/pointcloud/ tests/mesh/ -q
```
Expected: all pass. `tests/wrapper/test_reconstructor.py` carries a foreign diff — if it fails,
check whether the failure predates this change (`git stash` the foreign hunk is NOT allowed;
inspect and report instead).

- [ ] **Step 5: Format and commit**

**Do NOT run black or isort on `collab_splats/wrapper/reconstructor.py`.** Measured while
executing Task 12: this venv's black (26.5.1, newer than whatever last formatted the repo)
reformats 74 pre-existing lines of that file — `_STAGE_ORDER`, the undistort log call, two
`load_zarr` wraps — none of them yours. That churn buries the real change and touches lines
another session may be editing. Format the NEW test file only, and hand-match the surrounding
style (120 columns, trailing commas) in `reconstructor.py`. Verify with
`rtk proxy git diff --stat` that `reconstructor.py`'s line count matches what you actually
changed before committing.

```bash
/opt/venv/reconstruction/bin/python -m black --target-version py311 -l 120 tests/wrapper/test_vda_context.py
/opt/venv/reconstruction/bin/python -m isort tests/wrapper/test_vda_context.py
rtk proxy git diff --stat -- collab_splats/wrapper/reconstructor.py   # only your lines
git add tests/wrapper/test_vda_context.py
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_vda_context.py -m "feat(wrapper): wire vda_context_fps, depth_align, random_seed and splat_depth"
```

---

## Task 12: Config surface and docs — DONE (`90df852a`)

> **RUN THIS BEFORE TASK 11.** Task 11 indexes `self.config["preproc"]["vda_context_fps"]`
> directly, and `tests/wrapper/test_reconstructor.py:122` (`test_no_inline_defaults_in_source`)
> forbids a `.get("key", default)` fallback in `reconstructor.py` — base.yaml is the only
> permitted default source. So the yaml key must exist before the wiring that reads it.
> Task 10 hit this same landmine and resolved it by carrying the default on the function
> signature instead; Task 11 cannot, because its value is genuinely user-facing config.

**Files:**
- Modify: `configs/base.yaml` (FOREIGN DIFFS — stage with `git commit --only`)
- Modify: `configs/README.md`
- Modify: `collab_splats/wrapper/reconstructor.py` (the `mesh()` call site Task 10 deferred)
- Test: `tests/wrapper/test_reconstructor.py` (read-only; confirm no key-validation break)

**On the foreign `configs/base.yaml` diff:** that file carries another session's uncommitted
edits (`preproc.fps 1.0 -> 2.0`, and the whole `mesh:` block — `voxel_size`, `sdf_trunc`,
`depth_trunc`, `clean_repair`, `conf_percentile`, `native_resolution`,
`color_map_iterations`). `git commit --only configs/base.yaml` re-stages from the worktree and
would sweep all of it in. Capture the foreign hunks as a real patch FIRST — plain `git diff` is
rewritten by the RTK hook into a non-patch, so use `rtk proxy git diff -- configs/base.yaml >
foreign.patch` and confirm it with `git apply --check --reverse foreign.patch`. Then revert the
file, add only the new keys, commit, and re-apply the foreign patch.

- [x] **Step 1: Add the knobs to `configs/base.yaml`**

Under `preproc:`, after `undistort:`:

```yaml
  vda_context_fps: null       # sfm only: run VDA over a contiguous grid at this rate instead of
                              # over the 2-FPS keyframes, keeping only the keyframe rows. VDA is
                              # temporal, so a sparse subsample is out of distribution. Setting it
                              # also restricts keyframe selection (and blur substitution) to the
                              # same grid, so keyframes are grid members by construction.
                              # Measured 2026-08-26 at 8.56 FPS: per-frame depth CV 0.409 -> 0.401,
                              # co-visible sep-50 disagreement 34.5% -> 25.3%. Incompatible with
                              # frame_selection: optical_flow. null = keyframe-only VDA.
```

Under `pointcloud.instantsfm:`, after `retriangulation:`:

```yaml
    depth_align: scale        # scale | affine — how VDA metric depth is mapped to the COLMAP
                               # world. 'scale' fits one robust multiplier per frame. 'affine'
                               # fits 1/d_colmap ~= a*(1/d_vda) + b per frame, the space
                               # gsplat's depth_l1_loss is actually paid in; measured
                               # 2026-08-26 it cuts the irreducible depth-loss floor 18.9%
                               # pooled and 28.8% in the near field. 'affine' also writes 0
                               # (= no target) for pixels past its saturation horizon and past
                               # the furthest track observation — one-sided, far side only.
    random_seed: null          # seed InstantSfM's RUNTIME_OPTIONS (numpy/random/torch/cuda).
                               # InitializeRandomPositions draws unseeded, so two runs of the
                               # same scene differ. null = upstream behaviour (unseeded).
```

Under `mesh:`, after `color_map_iterations:`:

```yaml
  splat_depth: expected     # source: splats only — 'expected' (alpha-weighted rendered depth)
                            # or 'median' (RaDe-GS surface depth, 2dgs renders only).
```

Under `splats.losses.normal_consistency`, document the new spec key by changing the comment
block above `losses:` to:

```yaml
  # Each entry: {weight[, start, end, end_weight]} — with end the weight decays log-linearly
  # from weight at start to end_weight at end and holds there (e.g. fade the depth prior:
  # depth: {weight: 0.01, end: 12000, end_weight: 0.001}). normal_consistency additionally
  # accepts depth_ratio (2dgs only, default 0): the RaDe-GS blend weight on the median-depth
  # normal, (1-r)*cos(n, dn_expected) + r*cos(n, dn_median).
```

- [x] **Step 2: Verify the config loads and validates**

```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.wrapper.reconstructor import Reconstructor
from collab_splats.splats.trainer import SplatsConfig
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('configs/base.yaml').read_text())
print('preproc.vda_context_fps =', cfg['preproc']['vda_context_fps'])
print('depth_align =', cfg['pointcloud']['instantsfm']['depth_align'])
print('random_seed =', cfg['pointcloud']['instantsfm']['random_seed'])
print('mesh.splat_depth =', cfg['mesh']['splat_depth'])
SplatsConfig.from_dict(cfg['splats'])
print('SplatsConfig OK')
"
```
Expected: the four values print and `SplatsConfig OK`

- [x] **Step 3: Wire the `mesh()` call site Task 10 deferred**

Task 10 added `splat_depth` to `_run_tsdf_mesh`'s signature but deliberately did NOT add the
call-site line, because doing so before the yaml key existed would have meant a
`.get("splat_depth", "expected")` fallback, which `test_no_inline_defaults_in_source` rejects.
Now that Step 1 has added the key, add the line. In `collab_splats/wrapper/reconstructor.py`,
in `mesh()` (~`:1339`), find the `_run_tsdf_mesh(...)` call and add to its keyword arguments,
beside the sibling `conf_percentile=` / `clean_repair=` lines:

```python
            splat_depth=mesh_cfg["splat_depth"],
```

Direct indexing, not `.get()` — the key is guaranteed by base.yaml, and the architectural test
requires it. Without this line the yaml key is inert.

**Then pin the hop you just created.** Task 10's remediation (`9ec5c38f`) added
`test_run_tsdf_mesh_forwards_splat_depth_to_the_adapter`, which covers `_run_tsdf_mesh` -> the
splats adapter. It does NOT cover `mesh()` -> `_run_tsdf_mesh`, because that call-site line did
not exist when it was written. Deleting the line you just added would therefore still pass the
whole suite — the same silent-no-op defect the review caught one level down. Append to
`tests/wrapper/test_splats_stage.py`:

```python
def test_mesh_stage_forwards_splat_depth_from_the_config(tmp_path):
    """
    mesh.splat_depth has to survive the first hop too — config -> mesh() -> _run_tsdf_mesh.
    """
    # Patch the callee, not the adapter: this pins the config read and the keyword, and stays
    # green regardless of what the adapter does with the value
    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as run_mesh:
        recon = _reconstructor_with(tmp_path, mesh={"source": "splats", "splat_depth": "median"})
        recon.mesh()

    assert run_mesh.call_args.kwargs["splat_depth"] == "median"
```

Match the file's existing helper for building a Reconstructor with config overrides — copy the
shape from whichever sibling test already constructs one, rather than inventing
`_reconstructor_with` if it does not exist. Verify it has teeth by deleting the
`splat_depth=mesh_cfg["splat_depth"]` line and confirming this test, and only this test, fails.

- [x] **Step 4: Verify the unknown-spec-key error message names `depth_ratio`**

Moved into the Task 9 remediation commit (it edits `trainer.py`, which this task does not).
Verify only — the message for an unknown key on `normal_consistency` should name the legal set
including `depth_ratio`, not the hardcoded `{weight[, start, end, end_weight]}`:

```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.splats.trainer import SplatsConfig
try:
    SplatsConfig.from_dict({'primitive': '2dgs', 'losses': {'normal_consistency': {'weight': 0.05, 'depth_ration': 0.6}}})
except ValueError as e:
    print(e)
"
```
Expected: the message lists `depth_ratio` among the legal keys. If it does not, the Task 9
remediation did not land — fix it there, not here.

- [x] **Step 5: Document the knobs in `configs/README.md`**

Add one row/paragraph per knob in the sections that already document `preproc`,
`pointcloud.instantsfm`, `mesh`, and `splats.losses`, matching that file's existing format.
Each entry states: what it does, its default, and the measured justification (the numbers in the
yaml comments above).

- [x] **Step 6: Run the full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q -x --ignore=tests/integration`
Expected: no new failures against `docs/known-test-failures.md`

- [x] **Step 7: Commit (own hunks only — `configs/base.yaml` carries foreign work)**

```bash
git diff configs/base.yaml   # confirm only your hunks are staged-worthy; if foreign hunks are
                             # interleaved, commit with `git commit --only` and verify with
                             # `git show --stat HEAD`
git commit --only configs/base.yaml configs/README.md collab_splats/wrapper/reconstructor.py \
  tests/wrapper/test_splats_stage.py \
  -m "feat(configs): vda_context_fps, depth_align, random_seed, splat_depth"
```

---

## Task 13: Run the grid — 3 new cells against a reused baseline

**Files:**
- Create: `evals/results/2026-08-26-depth-align-grid/` (gitignored)

Do not start any cell until Tasks 1-12 are committed and the suite is green.

**Cell 1 is already on disk — do not re-run it.**
`/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist_r7/instantsfm/splats_combined2dgs/`
holds a 2dgs / 12k / 300-frame run at PSNR 20.805, SSIM 0.6776, 1,875,613 gaussians, 803.3 s,
with `ckpt.pt`, `mesh.ply` and `mesh_v0.1.ply` (the voxel-0.1 mesh, 422 MB) beside it. Its
config is reproduced verbatim as the common block below, so the other three cells differ from it
only in the knob under test. Keeping the `ckpt.pt` also means the mesh can be re-fused at other
parameters without retraining.

**Budget note:** the voxel-0.1 mesh on this scene took roughly 80 minutes (mesh.ply 03:51 ->
mesh_v0.1.ply 05:12), against ~13 minutes of training. Meshing, not training, is what this grid
costs — 6x per cell. Reusing cell 1 saves about 1.5 hours, and training all three before fusing
any of them (Steps 2-4) means the mesh budget is spent only where the PSNR table says it is
worth spending.

- [ ] **Step 1: Write the three override configs**

Create one yaml per new cell in the scratchpad. Common to all four cells — this is cell 1's
recorded config, not an idealised one:

```yaml
semantics: {enabled: false}
preproc: {undistort: true, max_frames: 300, search_radius: 7}
pointcloud:
  method: sfm
  backend: instantsfm
  instantsfm: {retriangulation: true}
mesh: {enabled: true, source: splats, voxel_size: 0.2, sdf_trunc: 0.8, depth_trunc: 100.0,
       clean_repair: true, conf_percentile: 0, native_resolution: true, color_map_iterations: 0}
splats:
  enabled: true
  primitive: 2dgs
  max_steps: 12000
  pose_opt: true
  appearance_opt: true
  grow_grad2d: 2.0e-4
  normalize_scene: true
  losses:
    depth: {weight: 0.01, end: 12000, end_weight: 0.001}
    normal_consistency: {weight: 0.05, start: 7000, depth_ratio: 0.0}
    distortion: {weight: 0.01, start: 3000}
    opacity_reg: {weight: 0.0}
    scale_reg: {weight: 0.0}
    appearance_reg: {weight: 0.001}
```

Two details in that block are load-bearing and are easy to "clean up" into a different
experiment:

- `depth` decays from 0.01 to 0.001 by step 12000. Cell 1 was trained that way, so every cell
  must be. It does mean the depth prior — the thing `depth_align: affine` improves — is weakest
  at the end of training, which biases the affine measurement *downward*. Read a positive affine
  result as a floor, not a ceiling.
- `opacity_reg` and `scale_reg` must be listed at `0.0` explicitly. `configs/base.yaml`
  regularisers survive the deep merge, so omitting them silently re-enables them and the run is
  no longer comparable to cell 1.

Per-cell deltas:

| cell | `preproc.vda_context_fps` | `instantsfm.depth_align` | `instantsfm.random_seed` | `normal_consistency.depth_ratio` | status |
|---|---|---|---|---|---|
| 1 | `null` | `scale` | `null` | `0.0` | ON DISK — reuse, do not run |
| 2 | `null` | `affine` | `null` | `0.0` | run |
| 3 | `null` | `affine` | `null` | `0.6` | run |
| 4 | `8.0` | `affine` | `0` | `0.6` | run |

Cells 2 and 3 must share cell 1's *reconstruction*, not merely its scene directory — and
neither setting of `--stages pointcloud` gives that. Measured while preparing the run:

- `build_pointcloud(overwrite=False)` short-circuits on `_stage_output_exists("pointcloud")`
  (`reconstructor.py:860`) and loads the model from disk, so `apply_depth_alignment` never runs
  and the zarr keeps cell 1's **scale**-aligned depth. The cell would silently be a rerun of
  cell 1.
- `build_pointcloud(overwrite=True)` reaches `_run_sfm`, and `InstantSfMCreator.reconstruct`
  deletes the whole `colmap/sparse/` tree before mapping (`sfm.py:1198-1206`). Unseeded, that
  puts fresh SfM noise on top of the depth-alignment change and the affine delta stops being
  attributable.

Re-align against the model already on disk instead. This is the `_run_sfm` tail
(`reconstructor.py:1076-1096`) verbatim, minus the mapping call, so it exercises the shipped
`apply_depth_alignment` path:

```python
model = pycolmap.Reconstruction(str(backend_dir / "colmap" / "sparse" / "0"))
store = FrameStore.open(recon.frames_zarr)
outputs = recon._sfm_result_from_reconstruction(model, backend_dir, store)
attrs = apply_depth_alignment(outputs, model, model="affine")
outputs.save_zarr(
    backend_dir / "pointcloud.zarr",
    extra_attrs={"method": "sfm", "backend": "instantsfm",
                 "instantsfm_version": importlib.metadata.version("instantsfm"), **attrs},
)
```

Then `recon.build_pointcloud()` (no overwrite — it now loads the model from disk) and
`recon.splats(overwrite=True)`.

Overwriting cell 1's `pointcloud.zarr` this way is safe: it is a pure function of `sparse/0`
plus the cached VDA npys, so re-running the same driver with `model="scale"` restores it, and
cell 1's finished outputs in `splats_combined2dgs/` never read it again. Rename `splats/` to
`splats_cell<N>/` after each cell so the next one starts clean.

Cell 4 is the only cell that legitimately re-runs SfM — its context stream changes frame
selection, so the reconstruction *must* differ. It sets `random_seed: 0` so that run is at
least reproducible, and it needs its own scene directory.

Write each cell's outputs to a fresh subdirectory so nothing overwrites
`splats_combined2dgs/`. Note that zarr `mode="w"` rmtree's a symlinked `splats.zarr`; a
directory-level symlink is safe, a file-level one is not.

- [ ] **Step 2: Train all three cells first — no meshing yet**

Training is ~13 min/cell against ~80 min/mesh, so all three train before anything is fused.
`ckpt.pt` is written at the end of training and meshing reads it, so deferring the fuse never
costs a retrain.

Override `mesh: {enabled: false}` on top of the common block for this pass and run
`--stages pointcloud splats`. The mesh is re-run later with a leaf-only `--stages mesh`, which
pulls its inputs from the processed scene rather than recomputing them.

One GPU job at a time — the container cgroup caps at 46.6 GB and parallel heavy jobs OOM.

```bash
SP=/tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad
tmux new-session -d -s grid_cell2 \
  '/opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py \
     --config '"$SP"'/cell2.yaml --stages pointcloud splats 2>&1 | tee '"$SP"'/cell2.log'
```

Wait for each session to exit before launching the next. Check `tmux list-sessions` and the tail
of the log rather than polling on a timer.

- [ ] **Step 3: Report PSNR/SSIM for all three cells**

Read `summary.psnr` / `summary.ssim` from each run's `splats_quality_report.json`. Cell 1 is
20.805 / 0.6776 and is not re-run. Present all four rows in one table.

All four sit below the 30k-step record because 12k gives 40 visits/view against 100; cross-cell
deltas are what matter, not the absolute level. Remember the depth prior decays to 0.001 by step
12000, so a positive `depth_align: affine` delta is a floor rather than a ceiling.

- [ ] **Step 4: GATE — decide which cells to mesh**

**Do not start any mesh until the Step 3 table has been reported and the choice of cells has
been made.** User directive, 2026-08-26: "report psnr for all first then decide meshing."

This is a gate, not a formality. Mesh quality is the primary objective and PSNR is only the
secondary signal, so PSNR cannot settle the question on its own — but at ~80 min per mesh,
fusing all three unconditionally costs about four hours of GPU time to answer a question three
of those hours may not be needed for. Propose a subset with reasoning; do not pick it
unilaterally.

Two things bias the decision away from "just mesh the winner":

- `depth_ratio` (cell 3) changes the *normal* supervision, which the TSDF fuse reads through
  the surface it converges to. Its PSNR delta may be ~0 while its mesh delta is not. A flat
  PSNR row is not evidence against meshing that cell.
- Cell 4 changes frame selection, so its reconstruction differs. Its PSNR is not comparable
  to cells 1-3 on equal terms.

- [ ] **Step 5: Mesh the chosen cells**

Leaf-only re-run against the already-trained scene:

```bash
tmux new-session -d -s mesh_cell2 \
  '/opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py \
     --config '"$SP"'/cell2.yaml --stages mesh 2>&1 | tee '"$SP"'/mesh_cell2.log'
```

One at a time, same OOM constraint. `mesh: {enabled: true, source: splats, voxel_size: 0.1,
conf_percentile: null}` must be restored in the config for this pass.

Note that zarr `mode="w"` rmtree's a symlinked `splats.zarr`; a directory-level symlink is safe,
a file-level one is not.

- [ ] **Step 6: Grade the meshes (primary)**

For each meshed cell, record: main-component vertex fraction, speckle component count, total
vertex count after `clean_repair`, and renders from the scene cameras. Vertex counts alone have
hidden truncation before — always render.

```bash
/opt/venv/reconstruction/bin/python - <<'PY'
import open3d as o3d, sys
mesh = o3d.io.read_triangle_mesh(sys.argv[1] if len(sys.argv) > 1 else "mesh.ply")
labels, counts, _areas = mesh.cluster_connected_triangles()
import numpy as np
counts = np.asarray(counts)
print("components:", len(counts), "main frac:", counts.max() / counts.sum(), "verts:", len(mesh.vertices))
PY
```

One `OffscreenRenderer` per process — open3d does not tolerate more.

Cell 1's reference numbers for 2dgs on this scene: main-component fraction 0.634 (against 0.457
for 3dgs), which is why 2dgs is the mesh source at all.

- [ ] **Step 7: Write the results document**

Create `docs/superpowers/specs/2026-08-26-depth-align-grid-results.md` with the four-cell table
(cell 1 reused), PSNR for all cells and mesh grades for the meshed subset, the verdict per
component, and which defaults (if any) should flip. Say explicitly which cells were not meshed
and why. Commit with `git add -f`.

---

## Follow-ups found during review (not in the original spec)

- **`weight` accepts a bool the same way `depth_ratio` used to.** Task 9's remediation
  (`f1443802`) type-checks `depth_ratio` but the sibling reads — `spec.get("weight", 0.0)` in
  `loss_weight`, and `distortion_spec.get("weight", 0.0) > 0` in the trainer's guard — have no
  type check at all. YAML parses `yes`/`on`/`true` to `True`, so `weight: yes` on any loss
  silently trains at 1.0. On `normal_consistency` that is 20x the intended 0.05, with no
  diagnostic. Same class of trap, one field over, and `weight` reaches arithmetic where
  `depth_ratio` only reached a comparison. Not blocking the grid (our override configs use
  unquoted floats), so it is deliberately deferred rather than folded into Task 9. Fix in one
  commit against `trainer.py` after Task 13, reusing the `isinstance(raw, bool) or not
  isinstance(raw, (int, float))` shape already there.

- **The no-grid window radius is derived from the first target gap only.** `spacing =
  targets[1] - targets[0]` in `_sample_by_quality`, so irregular gaps let neighbouring search
  windows overlap and two targets collapse onto one frame. Measured at the shipped
  `search_radius=7`: 17.3% of realistic `(total, n=30)` shapes overlap, 18.3% at `n=300`. The
  dedup pass added in `24761838` catches the consequence, so nothing is broken — but
  `min(np.diff(targets))` would remove the cause. Deliberately not folded into Task 3's
  remediation: it changes frame selection, and the grid runs must not move underneath the
  measurement. Both `parity_baseline.json` cases have `radius_min == radius_first == 3`, so the
  fix is verified not to move the parity baseline when it is eventually made.

- **A `splats.zarr` missing its `depth` array still fails with a bare `KeyError`.** Task 10's
  remediation (`9ec5c38f`) scoped the friendly missing-array message to the `median` branch,
  because the generic version told an `expected` user to "use splat_depth: expected". That
  restores exactly the pre-`8bf2988e` behaviour on the default path, so it is a faithful revert
  rather than an improvement. Making both paths fail actionably is a separate change — a generic
  message naming whichever array is absent, with the "needs a 2dgs run from this version"
  sentence appended only on the median branch. Not done deliberately; doing it naively re-creates
  the bug that was just fixed.

---

## Self-Review

**Spec coverage:**

| Spec component | Task(s) |
|---|---|
| A — `context_indices` | 1 |
| A — `decode_context` | 2 |
| A — `_sample_by_quality(candidates=)`, `sample_fps`/`sample_uniform` passthrough | 3 |
| A — `generate_vda_depth(keep_rows=)` | 4 |
| A — reconstructor wiring + video-absent fallback | 11 |
| B — `_depth_correspondences` extraction | 5 (Part A) |
| B — `align_depth_affine`, p99 positivity guard, MAD rejection, scale fallback | 5 (Part B) |
| B — `depth_align_model` / `depth_affine_ab` attrs, `depth_scale: "colmap"` retained | 6 |
| C — `median_depth` + `depth_normal_median` in the render | 8 (Part A) |
| C — uniform `spec` 5th arg, blended loss, default 0.0 | 9 (Part A) |
| C — trainer allow-list, range check, 2dgs-only | 9 (Part B) |
| C — `outputs.py` writes `median_depth` | 8 (Part B) |
| D — `random_seed` | 7, 11 |
| E — `mesh.splat_depth` | 10 |
| F — one-sided far bound | 5 (Part B, inside `align_depth_affine` / `_apply_affine_depth`) |
| F — honest sfm masking log | 11 |
| Config surface | 12 |
| Grid + grading | 13 |
| Comment fix at `sfm.py:307` (metric=True) | 4 (`_load_vda_model`) |

**Placeholder scan:** none — every code step carries the code, every command carries its
expected output. Two steps say "match the existing helper" (Task 8's `_toy_scene` and
`_render_scene`) because those files' fixture style must be read first; both name the exact
source (`tests/splats/synthetic.make_scene`) and the exact signature to check
(`trainer.py:222`).

**Type consistency:**
- `context_indices(...) -> list[int]` feeds `decode_context(video_path, indices, ...)` and
  `_context_keep_rows(grid, keyframe_indices)`; all three take/return plain `list[int]`.
- `align_depth_affine` returns `(coeffs (N,2) float64, far_limits (N,) float64, stats dict)`;
  `_apply_affine_depth(depth_row, a, b, far_limit)` consumes exactly those three scalars per
  row, and `apply_depth_alignment` is the only caller of both.
- `_depth_correspondences` returns `list[tuple[np.ndarray, np.ndarray]]` in `(d_colmap, d_vda)`
  order, consumed identically by `align_depth_to_reconstruction` and `align_depth_affine`.
- All six loss functions take `(render, target, gaussians, scene_scale, spec)`; `compute_losses`
  is the only caller inside the package, and the tests are updated in the same task.
- `splat_depth` is the parameter name in `_splats_to_tsdf_inputs`, `_run_tsdf_mesh`, and the
  `mesh:` yaml block — one spelling throughout.
- `depth_ratio` is spelled the same in `losses.py`, `trainer.py`, `base.yaml`, and every test.

**Known deviations from the spec, and why:**
- `decode_context` drops the spec's `roi` parameter: `undistort_frames` derives it
  deterministically from the profile, so accepting it separately would let a caller disagree
  with the crop the keyframes actually got.
- `align_depth_affine` returns `far_limits` as a third value rather than folding component F's
  mask into the coefficients, so the far bound stays inspectable and lands in zarr attrs.
