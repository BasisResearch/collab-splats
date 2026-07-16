# Dashboard UX Feedback Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dashboard explicit and responsive — every long operation shows step-level progress in the log window, no blocking work runs on the UI thread, busy state locks both tabs, and the frame slider live-previews.

**Architecture:** All heavy work already funnels through one `GpuWorker` daemon thread; this pass moves the stragglers there, adds an `OperationLog.step()` timing context manager used everywhere, and syncs a global busy state through the existing 300 ms op-log poll (poll, not push — one shared worker serves many browser sessions; pushing to per-session widgets from a shared object leaks dead sessions; polling matches the existing op_log architecture).

**Tech Stack:** Panel/Bokeh (single IOLoop per session), PyVista/VTK, matplotlib (Agg), rclone via `SessionSource`, ffmpeg via `collab_splats.preproc`.

**Spec:** `docs/superpowers/specs/2026-07-16-dashboard-ux-feedback-design.md`

**Environment:** run tests with `/opt/venv/reconstruction/bin/python -m pytest`. Format with `black . && isort .` before each commit. Commit docs/plan files with `git add -f`.

**Conventions in this repo:** flat test functions; block-level inline comments; `logging` not `print`; imports at top except heavy lazy imports (keep the existing lazy-import comments intact).

---

## File map

| File | Change |
|---|---|
| `collab_splats/dashboard/operation_log.py` | add `step()` ctx manager, `version` counter |
| `collab_splats/dashboard/async_utils.py` | add `on_error` callback to `run_off_loop` |
| `collab_splats/preproc/sampling.py` (+ `preproc/__init__.py`) | add `extract_frame_fast` (input-seek) |
| `collab_splats/dashboard/localize.py` | SceneCache bounded kinds; debounced preview; thumbnail; set_busy; worker-built figures; cached mesh; listing logs |
| `collab_splats/dashboard/shell.py` | deferred localize build + spinner; release_gpu → worker |
| `collab_splats/dashboard/app.py` | busy widened + poll sync; `_on_run` off-loop cache check + validation msg; single `_update_max_frames_bound`; step logging; listing logs |
| `collab_splats/dashboard/viewer.py` | lazy mesh via `ensure_mesh_polydata`; drop o3d re-read; op_log status lines |
| `collab_splats/dashboard/sources.py` | purge expired listing-cache entries |
| `collab_splats/dashboard/pipeline.py` | `step()` timings in `run_localization` |
| tests | `tests/dashboard/test_{operation_log,async_utils,localize_page,shell,app,viewer,sources}.py`, `tests/preproc/test_extract_frame_fast.py` |

---

### Task 1: `OperationLog.step()` + `version` counter

**Files:**
- Modify: `collab_splats/dashboard/operation_log.py`
- Test: `tests/dashboard/test_operation_log.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/dashboard/test_operation_log.py`)

```python
def test_step_logs_start_and_elapsed():
    log = OperationLog()
    with log.step("zarr read"):
        pass
    assert log.log_lines[0] == "zarr read…"
    assert log.log_lines[1].startswith("zarr read done (")
    assert log.log_lines[1].endswith("s)")


def test_step_logs_failure_and_reraises():
    log = OperationLog()
    with pytest.raises(ValueError):
        with log.step("mesh read"):
            raise ValueError("boom")
    assert log.log_lines[-1].startswith("mesh read FAILED (")
    assert "boom" in log.log_lines[-1]


def test_version_bumps_on_mutation_only():
    log = OperationLog()
    v0 = log.version
    log.append_line("a")
    v1 = log.version
    assert v1 > v0
    log.append_line("a")  # consecutive dupe is collapsed -> no bump
    assert log.version == v1
    log.start_op("x")
    log.update_progress(10, "y")
    log.finish_op()
    log.error_op("z")
    assert log.version > v1
```

Ensure the test file imports `pytest` at the top (add `import pytest` if absent).

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_operation_log.py -v`
Expected: FAIL — `AttributeError: 'OperationLog' object has no attribute 'step'`

- [ ] **Step 3: Implement**

In `operation_log.py`, add `import time` to the imports. In `__init__`, after `self._lock = threading.Lock()` add:

```python
        self._version = 0  # bumped on every visible mutation; UI polls compare-and-skip
```

Add a `version` property and the `step` context manager after `__init__`:

```python
    @property
    def version(self) -> int:
        """Monotonic change counter — pollers re-render only when it moves."""
        return self._version

    @contextlib.contextmanager
    def step(self, label: str):
        """Log '<label>…' on entry and '<label> done (Xs)' (or FAILED) on exit.

        Thread-safe and exception-safe; re-raises so callers still see failures.
        """
        self.append_line(f"{label}…")
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.append_line(f"{label} FAILED ({time.perf_counter() - t0:.1f}s): {exc}")
            raise
        else:
            self.append_line(f"{label} done ({time.perf_counter() - t0:.1f}s)")
```

Bump `_version` inside every mutating method's locked block: in `start_op`, `finish_op`, `error_op` add `self._version += 1` as the last line inside `with self._lock:`. In `update_progress` add it inside the lock. In `append_line`, add it after `self.log_lines = lines` (NOT on the collapsed-dupe early return).

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_operation_log.py -v`
Expected: all PASS (existing + 3 new)

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/operation_log.py tests/dashboard/test_operation_log.py && isort collab_splats/dashboard/operation_log.py tests/dashboard/test_operation_log.py
git add collab_splats/dashboard/operation_log.py tests/dashboard/test_operation_log.py
git commit -m "feat(dashboard): OperationLog.step timing context manager + version counter"
```

---

### Task 2: `run_off_loop` gains `on_error`

**Files:**
- Modify: `collab_splats/dashboard/async_utils.py`
- Test: `tests/dashboard/test_async_utils.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_on_error_called_with_exception():
    errors = []

    def fetch():
        raise RuntimeError("rclone down")

    t = run_off_loop(fetch, lambda r: None, label="x", doc=None, on_error=errors.append)
    t.join(timeout=5)
    assert len(errors) == 1
    assert "rclone down" in str(errors[0])


def test_error_without_handler_still_swallowed():
    t = run_off_loop(lambda: 1 / 0, lambda r: None, label="x", doc=None)
    t.join(timeout=5)  # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_async_utils.py -v`
Expected: FAIL — `TypeError: run_off_loop() got an unexpected keyword argument 'on_error'`

- [ ] **Step 3: Implement** — replace `run_off_loop` in `async_utils.py`:

```python
def run_off_loop(
    fetch: Callable[[], Any],
    apply: Callable[[Any], None],
    *,
    label: str,
    doc: Any | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> threading.Thread:
    """Run blocking `fetch()` on a daemon thread; marshal `apply(result)` back to the IOLoop.

    `doc` is captured on the IOLoop by the caller (pn.state.curdoc); None (tests / no server)
    runs `apply` inline. Exceptions in `fetch` are logged; `on_error(exc)` (if given) is
    marshalled back like `apply`, so callers can surface failures in the dashboard log.
    """

    def work() -> None:
        try:
            result = fetch()
        except Exception as exc:
            logger.warning("%s failed", label, exc_info=True)
            if on_error is not None:
                deliver_err = lambda e=exc: on_error(e)  # noqa: E731 — tiny marshal thunk
                doc.add_next_tick_callback(deliver_err) if doc is not None else deliver_err()
            return

        def deliver() -> None:
            apply(result)

        if doc is not None:
            doc.add_next_tick_callback(deliver)
        else:
            deliver()

    thread = threading.Thread(target=work, name=label, daemon=True)
    thread.start()
    return thread
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_async_utils.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/async_utils.py tests/dashboard/test_async_utils.py && isort collab_splats/dashboard/async_utils.py tests/dashboard/test_async_utils.py
git add collab_splats/dashboard/async_utils.py tests/dashboard/test_async_utils.py
git commit -m "feat(dashboard): run_off_loop on_error callback for surfacing fetch failures"
```

---

### Task 3: `extract_frame_fast` (ffmpeg input-seek)

`extract_frame` streams from frame 0 to N (`_iter_frames_at`) — exact but O(N) per call; a deep slider position would take seconds. The preview needs a fast approximate path; the localization run keeps the exact one.

**Files:**
- Modify: `collab_splats/preproc/sampling.py`, `collab_splats/preproc/__init__.py`
- Create: `tests/preproc/test_extract_frame_fast.py`

- [ ] **Step 1: Write the failing test**

```python
"""extract_frame_fast: input-seek single-frame decode for previews."""

import subprocess
import shutil

import numpy as np
import pytest

from collab_splats.preproc import extract_frame, extract_frame_fast

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    # 2s of 30fps testsrc — 60 frames, constant frame rate
    path = tmp_path_factory.mktemp("vid") / "synth.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=30",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_fast_matches_exact_shape_and_content(synth_video):
    fast = extract_frame_fast(synth_video, 30)
    exact = extract_frame(synth_video, 30)
    assert fast.shape == exact.shape == (240, 320, 3)
    assert fast.dtype == np.uint8
    # Same frame modulo codec noise: testsrc frames differ strongly frame-to-frame,
    # so a small mean error proves we seeked to the right frame.
    assert np.abs(fast.astype(int) - exact.astype(int)).mean() < 5


def test_fast_out_of_range_raises(synth_video):
    with pytest.raises(ValueError):
        extract_frame_fast(synth_video, 10_000)
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_extract_frame_fast.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_frame_fast'`

- [ ] **Step 3: Implement** — in `sampling.py`, directly after `extract_frame`:

```python
def extract_frame_fast(video_path: "str | Path", frame_idx: int) -> np.ndarray:
    """Decode one frame via ffmpeg input-seek; returns (H, W, 3) uint8 RGB.

    Seeks by timestamp (frame_idx / fps) before demuxing — O(1) in frame depth, so a
    deep frame previews instantly. Exact on constant-frame-rate video; may land one
    frame off near keyframes on VFR sources. Use extract_frame where exactness matters
    (e.g. the localization run, which records frame_idx as provenance).
    """
    _require_ffmpeg()
    info = get_video_info(str(video_path))
    fps, w, h, total = info["fps"], info["width"], info["height"], info["total_frames"]
    if not fps or not w or not h:
        # Unprobeable video: fall back to the exact streaming decode.
        return extract_frame(video_path, frame_idx)
    if total and frame_idx >= total:
        raise ValueError(f"extract_frame_fast: frame {frame_idx} past end of {video_path}")
    # -ss before -i = input seek (demuxer-level); rawvideo pipe avoids a temp file.
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{frame_idx / fps:.6f}", "-i", str(video_path),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    if len(raw) < w * h * 3:
        raise ValueError(f"extract_frame_fast: frame {frame_idx} not found in {video_path}")
    return np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()
```

In `collab_splats/preproc/__init__.py`, add `extract_frame_fast` to the imports from `.sampling` and to `__all__` (match the existing style of `extract_frame`).

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_extract_frame_fast.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/ tests/preproc/test_extract_frame_fast.py && isort collab_splats/preproc/ tests/preproc/test_extract_frame_fast.py
git add collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py tests/preproc/test_extract_frame_fast.py
git commit -m "feat(preproc): extract_frame_fast input-seek decode for dashboard previews"
```

---

### Task 4: SceneCache bounded kinds + listing-cache purge

**Files:**
- Modify: `collab_splats/dashboard/localize.py` (SceneCache), `collab_splats/dashboard/sources.py` (`_cached`)
- Test: `tests/dashboard/test_localize_page.py`, `tests/dashboard/test_sources.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_localize_page.py`:

```python
def test_scene_cache_evicts_oldest_mesh_beyond_keep():
    cache = SceneCache()
    for i in range(5):  # _KIND_KEEP["mesh"] == 3
        cache.put(("s", f"v{i}"), "mesh", f"m{i}")
    assert cache.get(("s", "v0"), "mesh") is None
    assert cache.get(("s", "v1"), "mesh") is None
    assert cache.get(("s", "v4"), "mesh") == "m4"


def test_scene_cache_unbounded_kinds_untouched():
    cache = SceneCache()
    for i in range(5):
        cache.put(("s", f"v{i}"), "localizer:disk", i)
    assert cache.get(("s", "v0"), "localizer:disk") == 0
```

(Ensure `SceneCache` is imported at the top of the test file; add it if absent.)

Append to `tests/dashboard/test_sources.py` (mirror the file's existing fake-client fixture style for constructing a `SessionSource` — reuse its existing mock/fake `RcloneClient` pattern):

```python
def test_listing_cache_purges_expired_entries(monkeypatch):
    src = SessionSource(client=object())  # client unused; producers are lambdas below
    src._listing_ttl = 0.0  # everything expires immediately
    src._cached(("a",), lambda: 1)
    src._cached(("b",), lambda: 2)
    src._cached(("c",), lambda: 3)  # refresh purges the expired a/b entries
    assert len(src._listing_cache) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py tests/dashboard/test_sources.py -v`
Expected: new tests FAIL (no eviction, cache grows to 5 / 3 entries)

- [ ] **Step 3: Implement**

In `localize.py`, add `from collections import deque` to the imports. Replace `SceneCache.__init__` and `put`:

```python
    # Kinds holding heavyweight objects get keep-last-N eviction; others are unbounded
    # ("loaded" is bounded by SplatsApp._remember_loaded; "localizer:*" by drop_kind).
    _KIND_KEEP = {"mesh": 3}

    def __init__(self) -> None:
        self._store: dict = {}
        self._order: dict[str, deque] = {}  # kind -> scene_key insertion order

    def put(self, scene_key, kind: str, value) -> None:
        self._store[(scene_key, kind)] = value
        keep = self._KIND_KEEP.get(kind)
        if keep is None:
            return
        order = self._order.setdefault(kind, deque())
        if scene_key in order:
            order.remove(scene_key)
        order.append(scene_key)
        while len(order) > keep:
            self._store.pop((order.popleft(), kind), None)
```

In `sources.py` `_cached`, after the cache-hit early return and before `value = producer()`:

```python
        # Refresh path: drop other expired entries so the memo doesn't grow unbounded.
        for k in [k for k, (exp, _v) in self._listing_cache.items() if exp <= now]:
            del self._listing_cache[k]
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py tests/dashboard/test_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/ tests/dashboard/ && isort collab_splats/dashboard/ tests/dashboard/
git add collab_splats/dashboard/localize.py collab_splats/dashboard/sources.py tests/dashboard/test_localize_page.py tests/dashboard/test_sources.py
git commit -m "perf(dashboard): bound SceneCache mesh entries; purge expired listing memos"
```

---

### Task 5: Deferred Localize build with spinner; release_gpu off the IOLoop

**Files:**
- Modify: `collab_splats/dashboard/shell.py`
- Test: `tests/dashboard/test_shell.py`

- [ ] **Step 1: Write the failing test** (append to `test_shell.py`; follow the file's existing pattern for constructing a `DashboardShell` with stub source/worker/op_log — reuse its fixtures)

```python
def test_localize_tab_shows_spinner_then_builds(shell_fixture_or_equivalent):
    """First activation immediately shows a building indicator; with no doc the build
    runs inline right after, replacing it with the real page."""
    shell = shell_fixture_or_equivalent
    shell.view()
    event = SimpleNamespace(new=1, old=0)
    shell._on_tab(event)
    # No server doc in tests -> deferred build ran inline; holder now holds the page.
    assert shell._localize_built
    assert len(shell._localize_holder) == 1
```

> Adapt the fixture name to what `test_shell.py` already uses (it constructs `DashboardShell` with mocked pages/worker). If it patches `LocalizePage.main`, assert the patched main's return object is in the holder. Add `from types import SimpleNamespace` if absent.

- [ ] **Step 2: Run to verify failure/behavior** — the assertion holds pre-change too (build was synchronous), so ALSO assert the op_log recorded the build:

```python
    assert any("localize page" in line for line in shell._op_log.log_lines)
```

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_shell.py -v`
Expected: FAIL on the log-line assertion

- [ ] **Step 3: Implement** — in `shell.py`, add `import time` to the imports, keep a shared op_log/worker reference in `__init__` (after the collaborators are resolved):

```python
        self._gpu = gpu_worker
        self._op_log = op_log
```

Replace `_on_tab`:

```python
    def _on_tab(self, event) -> None:
        """Swap sidebar to the active tab; build the localize view lazily; free GPU on leave."""
        if event.new == 1 and not self._localize_built:
            self._localize_built = True
            # Paint a spinner NOW; defer the heavy main() (pyvista/VTK) one tick so the
            # browser renders feedback before the build blocks the loop.
            self._localize_holder[:] = [
                pn.Column(
                    pn.indicators.LoadingSpinner(value=True, size=40),
                    pn.pane.HTML("<i>Building Localize page…</i>"),
                )
            ]

            def build() -> None:
                t0 = time.perf_counter()
                self._localize_holder[:] = [self._localize.main()]
                self._op_log.append_line(f"localize page built ({time.perf_counter() - t0:.1f}s)")

            doc = pn.state.curdoc
            doc.add_next_tick_callback(build) if doc is not None else build()
        page = self._splats if event.new == 0 else self._localize
        self._sidebar_holder[:] = [page.sidebar()]
        if event.old == 1:
            # pytorch_gc CUDA-syncs — run it on the worker, not the tab-switch watcher.
            self._gpu.submit(self._localize.release_gpu, lambda _res: None, pn.state.curdoc)
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_shell.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/shell.py tests/dashboard/test_shell.py && isort collab_splats/dashboard/shell.py tests/dashboard/test_shell.py
git add collab_splats/dashboard/shell.py tests/dashboard/test_shell.py
git commit -m "perf(dashboard): spinner + deferred localize build; release_gpu off the IOLoop"
```

---

### Task 6: Global busy sync across tabs

Poll-driven: each page's existing 300 ms `_tick` syncs its widgets from the shared `GpuWorker.busy`. Pages still disable immediately in their own click handlers (no 300 ms window on the initiating page).

**Files:**
- Modify: `collab_splats/dashboard/app.py` (`_set_busy`, `main()` tick), `collab_splats/dashboard/localize.py` (`set_busy`, `main()` tick, `_on_run`)
- Test: `tests/dashboard/test_app.py`, `tests/dashboard/test_localize_page.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_app.py` (mirror its existing SplatsApp construction fixture):

```python
def test_set_busy_disables_all_mutating_widgets(app_fixture):
    app = app_fixture
    app._set_busy(True)
    for w in (app.run_btn, app.force_btn, app.run_query_btn, app.view_mode,
              app.normalize_view, app.session_select, app.video_select):
        assert w.disabled
    assert "busy" in app.busy_note.object
    app._set_busy(False)
    assert not app.run_btn.disabled
    assert app.busy_note.object == ""


def test_sync_busy_follows_worker_flag(app_fixture):
    app = app_fixture
    app._gpu.busy = True
    app._sync_busy()
    assert app.run_btn.disabled
    app._gpu.busy = False
    app._sync_busy()
    assert not app.run_btn.disabled
```

Append to `tests/dashboard/test_localize_page.py` (mirror its LocalizePage fixture):

```python
def test_localize_set_busy_disables_widgets(page_fixture):
    page = page_fixture
    page.set_busy(True)
    for w in (page.run_btn, page.scene_session, page.scene_video, page.field_session,
              page.camera, page.query_video, page.method):
        assert w.disabled
    page.set_busy(False)
    assert not page.run_btn.disabled
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_localize_page.py -v`
Expected: FAIL — widgets not disabled / no `busy_note` / no `set_busy`

- [ ] **Step 3: Implement**

`app.py` — in `_build_sidebar`, after `self.force_btn = ...` add:

```python
        # Cross-tab busy indicator: filled while any GpuWorker job is in flight.
        self.busy_note = pn.pane.HTML("", sizing_mode="stretch_width")
```

and add `self.busy_note` to `self._sidebar` right after `pn.Row(self.run_btn, self.force_btn)`.

Replace `_set_busy` and add `_sync_busy`:

```python
    def _set_busy(self, busy: bool) -> None:
        """Enable/disable every mutating widget while a GPU job is in flight (IOLoop thread).

        Covers view widgets too: toggling view_mode mid-load would fire set_mode on a
        half-loaded viewer and queue a second job.
        """
        widgets = (
            self.run_btn, self.force_btn, self.run_query_btn, self.view_mode,
            self.normalize_view, self.session_select, self.video_select,
        )
        for w in widgets:
            w.disabled = busy
        op = self._op_log.current_op
        self.busy_note.object = (
            f"<span style='color:#e0a050;font-size:11px'>busy: {op or 'working'}…</span>" if busy else ""
        )

    def _sync_busy(self) -> None:
        """Poll hook: mirror the shared worker's busy flag onto this page's widgets."""
        busy = bool(self._gpu.busy)
        if busy != self.run_btn.disabled:
            self._set_busy(busy)
        elif busy:
            # Refresh the label while busy (current_op advances through the run).
            op = self._op_log.current_op
            self.busy_note.object = f"<span style='color:#e0a050;font-size:11px'>busy: {op or 'working'}…</span>"
```

In `main()`, extend `_tick` (and add version-gating — this is Task 6's half of the idle-poll fix; the render skip):

```python
        self._seen_log_version = -1

        def _tick() -> None:
            self._sync_busy()
            # Skip the HTML re-render when nothing changed (idle sessions poll for free).
            if self._op_log.version != self._seen_log_version:
                self._seen_log_version = self._op_log.version
                progress.object = self._op_log.render_html()
```

`localize.py` — add the same pattern. In `_build_sidebar` after `self.run_btn = ...`:

```python
        self.busy_note = pn.pane.HTML("", sizing_mode="stretch_width")
```

and add `self.busy_note` to `self._sidebar` after `self.run_btn`. Add methods (after `sidebar()`):

```python
    def set_busy(self, busy: bool) -> None:
        """Enable/disable this page's mutating widgets while a GPU job is in flight."""
        widgets = (
            self.run_btn, self.scene_session, self.scene_video, self.field_session,
            self.camera, self.query_video, self.method, self.append_db,
        )
        for w in widgets:
            w.disabled = busy
        op = self._op_log.current_op
        self.busy_note.object = (
            f"<span style='color:#e0a050;font-size:11px'>busy: {op or 'working'}…</span>" if busy else ""
        )

    def _sync_busy(self) -> None:
        """Poll hook: mirror the shared worker's busy flag onto this page's widgets."""
        busy = bool(self._gpu.busy)
        if busy != self.run_btn.disabled:
            self.set_busy(busy)
```

In `main()`, replace `_tick` with the synced, version-gated version:

```python
        self._seen_log_version = -1

        def _tick() -> None:
            self._sync_busy()
            if self._op_log.version != self._seen_log_version:
                self._seen_log_version = self._op_log.version
                self._progress.object = self._op_log.render_html()
```

In `_on_run`, replace `self.run_btn.disabled = True` with `self.set_busy(True)` and in `on_done` replace `self.run_btn.disabled = False` with `self.set_busy(False)`.

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_localize_page.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/ tests/dashboard/ && isort collab_splats/dashboard/ tests/dashboard/
git add collab_splats/dashboard/app.py collab_splats/dashboard/localize.py tests/dashboard/test_app.py tests/dashboard/test_localize_page.py
git commit -m "feat(dashboard): global busy lock across tabs via worker-flag poll; version-gated log render"
```

---

### Task 7: Debounced live frame preview + thumbnail

**Files:**
- Modify: `collab_splats/dashboard/localize.py`
- Test: `tests/dashboard/test_localize_page.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_frame_slider_preview_latest_wins(page_fixture, monkeypatch):
    page = page_fixture
    shown = []
    monkeypatch.setattr(page, "_ensure_local_query_video", lambda *a: Path("/dev/null"))
    monkeypatch.setattr(
        "collab_splats.preproc.extract_frame_fast",
        lambda video, idx: np.full((4, 4, 3), idx, dtype=np.uint8),
    )
    monkeypatch.setattr(page, "_show_frame", lambda f: shown.append(int(f[0, 0, 0])))
    page.field_session.options = ["fs"]; page.field_session.value = "fs"
    page.camera.options = ["rgb_0"]; page.camera.value = "rgb_0"
    page.query_video.options = ["v.mp4"]; page.query_video.value = "v.mp4"
    shown.clear()  # ignore the on-select frame-0 preview
    # Two quick slider moves: only the second may render (token supersedes the first).
    page._preview_frame(token=1, frame_idx=5, doc=None)   # stale token
    page._preview_token = 2
    page._preview_frame(token=2, frame_idx=9, doc=None)   # current token
    assert shown == [9]


def test_show_frame_downscales_to_thumbnail(page_fixture):
    page = page_fixture
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    page._show_frame(big)
    assert page._frame_pane.object.width <= 640
```

Wait — `_preview_frame(token=1, ...)` with `page._preview_token` defaulting to 0 would skip both. Set `page._preview_token = 1` before the first call, then bump to 2 before the second call **and call the first again after** to prove staleness:

```python
    page._preview_token = 2
    page._preview_frame(token=1, frame_idx=5, doc=None)   # superseded -> dropped
    page._preview_frame(token=2, frame_idx=9, doc=None)   # current -> shown
    assert shown == [9]
```

Use this corrected form in the test file. Add `from pathlib import Path` / `import numpy as np` if not already imported there.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_preview_frame'`

- [ ] **Step 3: Implement**

In `localize.py` add `import time` to the imports and two constants after `_SUBSAMPLE_ABOVE`:

```python
_PREVIEW_DEBOUNCE_S = 0.3  # slider settles this long before a frame decode fires
_PREVIEW_MAX_W = 640  # thumbnail width pushed to the browser (full-res is wasteful)
```

In `__init__`, before `self._build_sidebar()`:

```python
        self._preview_token = 0  # latest slider request; stale extracts are dropped
        self._preview_timer: threading.Timer | None = None
```

In `_build_sidebar`, wire the watcher (with the others):

```python
        self.frame_slider.param.watch(self._on_frame_slider, "value")
```

Add after `_on_query_video`:

```python
    def _on_frame_slider(self, event) -> None:
        """Debounced live preview: decode + show the frame shortly after the slider settles."""
        if self._gpu.busy:
            return  # a run owns the panes; the slider still sets the run's frame_idx
        if not (self.field_session.value and self.camera.value and self.query_video.value):
            return
        self._preview_token += 1
        if self._preview_timer is not None:
            self._preview_timer.cancel()
        self._preview_timer = threading.Timer(
            _PREVIEW_DEBOUNCE_S,
            self._preview_frame,
            kwargs={"token": self._preview_token, "frame_idx": event.new, "doc": pn.state.curdoc},
        )
        self._preview_timer.daemon = True
        self._preview_timer.start()

    def _preview_frame(self, token: int, frame_idx: int, doc) -> None:
        """Timer thread: fast-seek decode, then marshal display back to the IOLoop."""
        if token != self._preview_token:
            return  # superseded by a newer slider position
        fs, cam, name = self.field_session.value, self.camera.value, self.query_video.value
        t0 = time.perf_counter()
        try:
            video = self._ensure_local_query_video(fs, cam, name)
            from collab_splats.preproc import extract_frame_fast

            frame = extract_frame_fast(video, frame_idx)
        except Exception as exc:
            logger.warning("frame preview failed", exc_info=True)
            self._op_log.append_line(f"frame {frame_idx} preview FAILED: {exc}")
            return
        if token != self._preview_token:
            return
        elapsed = time.perf_counter() - t0

        def show() -> None:
            if token != self._preview_token:
                return
            self._show_frame(frame)
            self._op_log.append_line(f"frame {frame_idx} loaded ({elapsed:.1f}s)")

        doc.add_next_tick_callback(show) if doc is not None else show()
```

Replace `_show_frame`:

```python
    def _show_frame(self, frame: np.ndarray) -> None:
        """Show the selected query frame in the left panel, downscaled to a thumbnail."""
        from PIL import Image as PILImage

        # pn.pane.Image renders PIL images directly; full-res frames push MBs of base64
        # into the doc, so cap the preview width (display is scale_width anyway).
        img = PILImage.fromarray(frame)
        if img.width > _PREVIEW_MAX_W:
            img = img.resize((_PREVIEW_MAX_W, max(1, int(img.height * _PREVIEW_MAX_W / img.width))))
        self._frame_pane.object = img
        self._matches_col[:] = [self._frame_pane]
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/localize.py tests/dashboard/test_localize_page.py && isort collab_splats/dashboard/localize.py tests/dashboard/test_localize_page.py
git add collab_splats/dashboard/localize.py tests/dashboard/test_localize_page.py
git commit -m "feat(dashboard): debounced live frame preview with thumbnail + log feedback"
```

---

### Task 8: `_on_run` off-loop cache check + validation; single frame-bound update

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_on_run_without_selection_logs_error(app_fixture):
    app = app_fixture
    app.session_select.options = []; app.video_select.options = []
    app._on_run(None, force=False)
    assert any("select a session" in line for line in app._op_log.log_lines)


def test_on_run_remote_check_runs_off_loop(app_fixture, monkeypatch):
    """has_processed must not be called synchronously inside the click handler."""
    app = app_fixture
    called_inline = []
    monkeypatch.setattr(app._source, "has_processed", lambda *a: called_inline.append(a) or True)
    loads = []
    monkeypatch.setattr(app, "_load_outputs", lambda s, st: loads.append((s, st)))
    app.session_select.options = ["s"]; app.session_select.value = "s"
    app.video_select.options = ["v.mp4"]; app.video_select.value = "v.mp4"
    calls_before = len(called_inline)
    app._on_run(None, force=False)
    # doc=None in tests -> run_off_loop's thread still does the check; join it.
    app._cache_check_thread.join(timeout=5)
    assert loads == [("s", "v")]
    assert len(called_inline) == calls_before + 1
```

> Adapt fixture setup to `test_app.py`'s existing SplatsApp fixture (it already stubs `SessionSource`; the `_on_video` watcher may fire `_autoload_current` — set `app._suppress_autoload = True` around the value assignments if the existing tests do so).

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v`
Expected: FAIL — no error log line; no `_cache_check_thread`

- [ ] **Step 3: Implement**

Replace `_on_run` and add `_start_run` in `app.py`:

```python
    def _on_run(self, event, force: bool) -> None:
        """Run or reload the pipeline, respecting cache and force flag."""
        # Flush pending UI state so the config driving this run is durable on disk.
        self._flush_state()
        session, name = self.session_select.value, self.video_select.value
        if not session or not name:
            self._op_log.error_op("select a session and a video first")
            return
        stem = Path(name).stem
        out = self._base_dir / session / stem
        if force:
            # Force re-run: drop cached loads so the post-run load re-reads fresh outputs.
            self._invalidate_scene(session, stem)
            self._start_run(session, name, stem)
            return
        if (out / "feedforward.zarr").exists():
            self._load_outputs(session, stem)
            return
        # Remote-cache check is a blocking rclone list — off the IOLoop (a cold check
        # inside the click handler froze the whole page), then load or run on the result.
        self._op_log.append_line(f"checking server for {stem}…")
        self._cache_check_thread = run_off_loop(
            lambda: self._source.has_processed(session, stem),
            lambda ok: self._load_outputs(session, stem) if ok else self._start_run(session, name, stem),
            label="has-processed",
            doc=pn.state.curdoc,
            on_error=lambda exc: self._op_log.error_op(f"server check failed: {exc}"),
        )

    def _start_run(self, session: str, name: str, stem: str) -> None:
        """Enqueue the full pipeline for a video (IOLoop thread)."""
        config = self._current_config()
        doc = pn.state.curdoc

        def job():
            # Lazy import: pulls the heavy reconstruction/mesh stack only when a run starts.
            from collab_splats.dashboard.pipeline import run_pipeline

            video = self._ensure_local_video(session, name)
            run_pipeline(
                video_path=video,
                session=session,
                stem=stem,
                config=config,
                op_log=self._op_log,
                source=self._source,
                base_dir=self._base_dir,
            )
            return True

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._invalidate_scene(session, stem)  # fresh outputs -> stale cache/display
            self._load_outputs(session, stem)  # re-enqueues a load job

        self._set_busy(True)
        self._op_log.start_op(f"running {stem}")
        self._gpu.submit(job, on_done, doc)
```

Fix the double frame-bound update + inline ffprobe. In `_on_video`, delete the line `self._update_max_frames_bound(self.session_select.value, event.new)` (`_autoload_current` already calls it — the duplicate spawned a second video download on remote scenes). Replace `_update_max_frames_bound` (delete its old body AND `_apply_max_frames_bound`'s ffprobe-on-loop call path) with:

```python
    def _update_max_frames_bound(self, session: str, name: str) -> None:
        """Set the Max-frames bound to the video's frame count — fetch/probe off the IOLoop.

        ffprobe is a subprocess even for local files; never run it inline in a watcher.
        """
        if not session or not name:
            return

        def work() -> int:
            video = self._ensure_local_video(session, name)  # no-op when already local
            from collab_splats.preproc import get_video_info

            return int(get_video_info(str(video)).get("total_frames") or 0)

        def apply(total: int) -> None:
            if total > 0:
                self.max_frames.end = total
                self.max_frames.name = f"Max frames (video has {total})"

        run_off_loop(
            work,
            apply,
            label="video-meta",
            doc=pn.state.curdoc,
            on_error=lambda exc: self._op_log.append_line(f"frame count unavailable: {exc}"),
        )
```

Delete `_apply_max_frames_bound` and update any tests referencing it to call `_update_max_frames_bound` (check `test_app.py` for usages and adapt them — the off-loop thread runs inline-ish with `doc=None`? No: `run_off_loop` always uses a thread; tests must join via the returned thread. If tests need it, have `_update_max_frames_bound` store the thread: `self._video_meta_thread = run_off_loop(...)`, and join in tests.) Use `self._video_meta_thread = run_off_loop(...)`.

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v`
Expected: PASS (adapt any pre-existing `_apply_max_frames_bound` tests as above)

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/app.py tests/dashboard/test_app.py && isort collab_splats/dashboard/app.py tests/dashboard/test_app.py
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "fix(dashboard): off-loop remote-cache check in Run; single frame-bound probe off-loop"
```

---

### Task 9: Step logging in load path + listing feedback

**Files:**
- Modify: `collab_splats/dashboard/app.py`, `collab_splats/dashboard/localize.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_outputs_logs_steps(app_fixture, tmp_path, monkeypatch):
    """A cold load logs pull/read steps with elapsed times (doc=None -> job runs inline)."""
    app = app_fixture
    # Fake a remote scene: no local zarr; pull_processed creates it; load_zarr stubbed.
    ...  # follow test_app.py's existing _load_outputs test setup (it stubs FeedforwardResult)
    app._load_outputs("s", "v")
    joined = "\n".join(app._op_log.log_lines)
    assert "pulling from server" in joined
    assert "reading feedforward.zarr" in joined and "done (" in joined
```

> `test_app.py` already tests `_load_outputs` with a stubbed `FeedforwardResult.load_zarr` — extend that existing test (or copy its arrangement) rather than inventing a new stub scheme. The assertion block above is the required delta.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v`
Expected: FAIL — no "reading feedforward.zarr" line

- [ ] **Step 3: Implement**

In `app.py` `_load_outputs`'s `job()`, wrap the two slow blocks and log the cache hit:

```python
            cached = self._cache.get((session, stem), "loaded")
            if cached is not None:
                self._op_log.append_line(f"{stem}: served from session cache")
                return cached
            if not (out / "feedforward.zarr").exists():
                with self._op_log.step(f"{stem}: pulling from server"):
                    self._source.pull_processed(
                        session,
                        stem,
                        out,
                        excludes=PULL_EXCLUDES,
                        on_line=self._op_log.rclone_progress("⬇ pulling from server"),
                    )
            with self._op_log.step(f"{stem}: reading feedforward.zarr"):
                result = FeedforwardResult.load_zarr(
                    out / "feedforward.zarr",
                    load_depth=False,
                    load_world_points=False,
                    load_confidence=False,
                    load_features=False,
                    load_pixel_indices=False,
                )
```

(rest of `job()` unchanged).

Listing feedback — wrap every rclone listing fetch in a `step`. Pattern (apply to each):

`app.py` `_refresh_sessions` `work()`:

```python
            try:
                with self._op_log.step("listing sessions"):
                    names = self._source.list_sessions()
            except Exception as exc:
                logger.warning("session listing failed: %s", exc)
                self._op_log.error_op(f"session listing failed: {exc}")
                names = []
```

`app.py` `_on_session`:

```python
        def fetch():
            with self._op_log.step(f"listing videos ({session})"):
                return self._source.list_videos(session)

        self._video_list_thread = run_off_loop(
            fetch,
            lambda vids: setattr(self.video_select, "options", vids),
            label="video-list",
            doc=pn.state.curdoc,
        )
```

(the `step` FAILED line already surfaces errors — no separate on_error needed where a step wraps the fetch).

`localize.py` — same pattern for `_refresh_listings` (wrap `list_sessions` → "listing scene sessions", `list_field_sessions` → "listing field sessions", and call `self._op_log.error_op(...)` in the except branches), `_on_scene_session` ("listing scene videos"), `_on_scene_video` ("listing feature DBs"), `_on_field_session` ("listing cameras"), `_on_camera` ("listing camera videos"), `_on_query_video` (wrap the fetch+probe in `step(f"fetching query video {name}")`, and in its except branch add `self._op_log.append_line(f"query video preview FAILED: see server log")` → better: capture `exc` and log `f"query video preview FAILED: {exc}"`).

`app.py` `max_display_points` watcher (line ~177) — replace the lambda:

```python
        def _on_density(event) -> None:
            # Bust the reselect short-circuit; density applies on the next scene (re)load.
            self._current_scene = None
            self._op_log.append_line(f"display density {event.new:,} — reselect the scene to apply")

        self.max_display_points.param.watch(_on_density, "value")
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_localize_page.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/ tests/dashboard/ && isort collab_splats/dashboard/ tests/dashboard/
git add collab_splats/dashboard/app.py collab_splats/dashboard/localize.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): step-level load logging + listing feedback in the op log"
```

---

### Task 10: Localize result rendering off the IOLoop + run stage timings

**Files:**
- Modify: `collab_splats/dashboard/localize.py`, `collab_splats/dashboard/pipeline.py`
- Test: `tests/dashboard/test_localize_page.py`, `tests/dashboard/test_run_localization.py`

- [ ] **Step 1: Write the failing test** (append to `test_localize_page.py`)

```python
def test_build_result_figures_is_pure(page_fixture, monkeypatch):
    """Figure building must be callable off the IOLoop: takes output, returns figs, touches no panes."""
    page = page_fixture
    out = make_fake_run_output()  # follow test_run_localization.py's LocalizationRunOutput fixture
    figs = page._build_result_figures(out, LocalizationConfig(extractor="disk"))
    assert set(figs) == {"dist_fig", "match_figs", "stats_html"}
```

> `test_run_localization.py` builds `LocalizationRunOutput` instances — reuse/extract its fixture. Monkeypatch `collab_splats.localization.viz.plot_inlier_distribution` / `plot_correspondences` to return `matplotlib.figure.Figure()` stubs so the test stays light.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -v`
Expected: FAIL — no `_build_result_figures`

- [ ] **Step 3: Implement**

Split `_render_result` into worker-safe figure building and IOLoop-only pane assignment.

Add to `localize.py`:

```python
    def _build_result_figures(self, out, config: LocalizationConfig) -> dict:
        """Build all matplotlib figures + stats HTML for a run output (worker thread — pure).

        Safe off the IOLoop: matplotlib Agg figures only, no pane/widget access. The
        GpuWorker serializes jobs, so pyplot's global state is never touched concurrently.
        """
        from collab_splats.localization.viz import plot_correspondences, plot_inlier_distribution

        loc = out.result
        n_frames = len(out.ref_image_paths)
        dist_fig = plot_inlier_distribution(loc, n_frames=n_frames, frame_sources=out.frame_sources)
        ratio = 100 * loc.n_inliers / max(loc.n_correspondences, 1)
        pose_msg = "" if loc.pose is not None else " — <b style='color:#e05050'>POSE FAILED</b>"
        stats_html = (
            f"<div style='font-size:12px'>inliers {loc.n_inliers}/{loc.n_correspondences} "
            f"({ratio:.0f}%) · intrinsics: {out.intrinsics_source} "
            f"(fx={out.query_intrinsics[0, 0]:.0f}){pose_msg}</div>"
        )
        # Top-k match-pair figures, best-first
        match_figs = []
        if loc.ref_frame_indices is not None and loc.inlier_mask is not None:
            counts = np.bincount(loc.ref_frame_indices[loc.inlier_mask].astype(np.intp), minlength=n_frames)
            top = np.argsort(counts)[::-1][: config.top_k_viz]
            for ref in top:
                if counts[ref] == 0 or not Path(out.ref_image_paths[ref]).exists():
                    continue
                mfig = plot_correspondences(
                    loc, out.query_frame, out.ref_image_paths,
                    max_pairs=config.max_pairs, ref_idx=int(ref), show=False,
                )
                if mfig is not None:
                    match_figs.append(mfig)
        return {"dist_fig": dist_fig, "match_figs": match_figs, "stats_html": stats_html}

    def _ensure_scene_mesh(self, scene_key, mesh_path: Path):
        """Read the scene mesh with cache (worker thread — pv.read is a blocking disk read)."""
        mesh = self._cache.get(scene_key, "mesh")
        if mesh is None and mesh_path.exists():
            with self._op_log.step("reading scene mesh"):
                mesh = pv.read(str(mesh_path))
            self._cache.put(scene_key, "mesh", mesh)
        return mesh
```

In `_on_run`, extend `job()` to build figures + read the mesh on the worker:

```python
        scene_key = (scene_session, stem)
        mesh_path = self._base_dir / scene_session / stem / "mesh" / "mesh_tsdf.ply"

        def job():
            # Lazy import: pulls the heavy stack only when a run starts (mirrors SplatsApp)
            from collab_splats.dashboard.pipeline import run_localization

            video = self._ensure_local_query_video(fs, cam, name)
            out = run_localization(
                query_video=video, frame_idx=frame_idx, session=scene_session, stem=stem,
                config=config, op_log=self._op_log, source=self._source,
                base_dir=self._base_dir, provenance=provenance, cache=self._cache,
            )
            # Figures + mesh read are slow — build them here so on_done only assigns panes.
            with self._op_log.step("building result figures"):
                figs = self._build_result_figures(out, config)
            mesh = self._ensure_scene_mesh(scene_key, mesh_path)
            return (out, figs, mesh)

        def on_done(res):
            self.set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            out, figs, mesh = res
            self._render_result(out, figs, mesh, scene_key)
```

(the local variable is `scene_session = self.scene_session.value` — it's already read at the top of `_on_run`.)

Rewrite `_render_result` to consume prebuilt figures (IOLoop-only pane work):

```python
    def _render_result(self, out, figs: dict, mesh, scene_key) -> None:
        """Assign prebuilt figures/mesh to the panes (IOLoop thread — no heavy work here)."""
        import matplotlib.pyplot as plt

        try:
            # Close the outgoing figures before replacing them so they don't accumulate
            # in pyplot's global registry (the pre-run frame pane is an Image).
            for child in list(self._matches_col):
                if isinstance(child, pn.pane.Matplotlib) and child.object is not None:
                    plt.close(child.object)
            old_dist = self._dist_pane.object
            self._dist_pane.object = figs["dist_fig"]
            if old_dist is not None:
                plt.close(old_dist)
            self._stats.object = figs["stats_html"]
            if figs["match_figs"]:
                self._matches_col[:] = [
                    pn.pane.Matplotlib(f, sizing_mode="stretch_width", tight=True) for f in figs["match_figs"]
                ]
            self._render_scene(mesh, out.ref_extrinsics, out.result.pose)
        except Exception as exc:
            logger.warning("localize render failed", exc_info=True)
            self._op_log.error_op(str(exc))
```

Rewrite `_render_scene` to take the preloaded mesh (drop its `pv.read`; delete the old cache block):

```python
    def _render_scene(self, mesh, extrinsics: np.ndarray, localized_pose: "np.ndarray | None") -> None:
        """Rebuild the 3D pane: preloaded mesh, time-coloured cameras, red localized camera."""
        self._ensure_plotter()
        self._plotter.clear()
        if mesh is not None:
            self._plotter.add_mesh(mesh, rgb="RGB" in mesh.array_names, opacity=0.9)
        # ... (camera-centers block onward is UNCHANGED from the current implementation:
        #      centers/step/sub/poly/add_mesh/add_text/localized red point/reset_camera/synchronize)
```

Keep the camera-plotting half of the current `_render_scene` verbatim.

`pipeline.py` `run_localization` — add elapsed-time steps around the three slow stages (keep the existing `update_progress` calls for the % bar):

```python
            op_log.update_progress(15, "localize: loading reconstruction")
            with op_log.step("localize: loading reconstruction"):
                result = _load_feedforward_result(out_dir)

            op_log.update_progress(25, f"localize: loading DB ({config.extractor})")
            with op_log.step(f"localize: DB ({config.extractor})"):
                localizer = _build_localizer(
                    result, config, out_dir / "feedforward.zarr", op_log, cache=cache, scene_key=(session, stem)
                )

            op_log.update_progress(70, "localize: matching + solving pose")
            with op_log.step("localize: matching + solving pose"):
                loc = localizer.localize(frame, K)
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py tests/dashboard/test_run_localization.py -v`
Expected: PASS (adapt any existing `_render_result`/`_render_scene` signature tests)

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/ tests/dashboard/ && isort collab_splats/dashboard/ tests/dashboard/
git add collab_splats/dashboard/localize.py collab_splats/dashboard/pipeline.py tests/dashboard/test_localize_page.py tests/dashboard/test_run_localization.py
git commit -m "perf(dashboard): build localize figures + read mesh on the worker; stage timings"
```

---### Task 11: Viewer — lazy mesh load, no o3d re-read, surfaced status

**Files:**
- Modify: `collab_splats/dashboard/viewer.py`, `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_viewer.py`, `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing tests** (append to `test_viewer.py`; use its existing off_screen fixtures/fake results)

```python
def test_load_does_not_read_mesh_eagerly(viewer_fixture, fake_result, tmp_path, monkeypatch):
    reads = []
    monkeypatch.setattr("pyvista.read", lambda p: reads.append(p) or pv.PolyData())
    mesh_path = tmp_path / "mesh_tsdf.ply"; mesh_path.touch()
    viewer_fixture.load(fake_result, mesh_path=mesh_path)
    assert reads == []  # pointcloud mode: mesh stays on disk until first mesh-mode switch


def test_ensure_mesh_polydata_reads_once_and_reports(viewer_fixture, fake_result, tmp_path, monkeypatch):
    reads = []
    monkeypatch.setattr("pyvista.read", lambda p: reads.append(p) or pv.PolyData())
    mesh_path = tmp_path / "mesh_tsdf.ply"; mesh_path.touch()
    viewer_fixture.load(fake_result, mesh_path=mesh_path)
    assert viewer_fixture.ensure_mesh_polydata() is True
    assert viewer_fixture.ensure_mesh_polydata() is True  # cached, no second read
    assert len(reads) == 1


def test_mesh_mode_without_mesh_logs(viewer_with_oplog_fixture, fake_result):
    viewer, op_log = viewer_with_oplog_fixture
    viewer.load(fake_result, mesh_path=None)
    viewer.set_mode("mesh")
    assert any("mesh not found" in line for line in op_log.log_lines)
```

> `viewer_with_oplog_fixture`: construct `SplitViewer(off_screen=True, op_log=OperationLog())` and return both. Adapt names to the file's existing fixture conventions.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -v`
Expected: FAIL — eager `pv.read` fires in `load`; no `op_log` kwarg

- [ ] **Step 3: Implement**

`viewer.py`:

1. `__init__(self, off_screen: bool = False, op_log=None)` — store `self._op_log = op_log`. Add a helper:

```python
    def _log(self, message: str) -> None:
        """Surface a viewer status line in the dashboard log (no-op without an op_log)."""
        if self._op_log is not None:
            self._op_log.append_line(message)
```

2. In `load()`, replace the two eager reads:

```python
        self._mesh_path = Path(mesh_path) if mesh_path else None
        # Mesh + vertex features load lazily on the first mesh-mode switch (worker thread
        # via ensure_mesh_polydata) — the default pointcloud view never pays the disk read.
        self._mesh_polydata = None
        self._mesh_vertex_features = None
```

3. Add after `load()`:

```python
    def ensure_mesh_polydata(self, preloaded: "pv.PolyData | None" = None, op_log=None) -> bool:
        """Read mesh + vertex features from disk if not yet loaded (call on the worker).

        preloaded lets the app hand in a SceneCache hit so the ply is read once per
        scene across both pages. Returns True when a mesh is available to render.
        """
        if self._mesh_polydata is not None:
            return True
        if preloaded is not None:
            self._mesh_polydata = preloaded
        elif self._mesh_path and self._mesh_path.exists():
            log = op_log or self._op_log
            if log is not None:
                with log.step("reading mesh"):
                    self._mesh_polydata = pv.read(str(self._mesh_path))
            else:
                self._mesh_polydata = pv.read(str(self._mesh_path))
        else:
            return False
        self._mesh_vertex_features = load_mesh_vertex_features(self._mesh_path.parent)
        return True
```

4. In `_render_left`, replace the silent fallback branch:

```python
            if self.mode == "mesh":
                self._status = "mesh not found."
                self._log("mesh not found — showing pointcloud")
                logger.warning("mesh not found; falling back to pointcloud for left pane")
```

5. In `ensure_mesh_features`, drop the o3d re-read — the PolyData already holds the vertices:

```python
        if self._mesh_vertex_features is not None:
            return
        if self._lifted_normed is None or self._mesh_polydata is None:
            return
        from collab_splats.mesh.utils import features2vertex

        if op_log is not None:
            op_log.append_line("query: transferring features to mesh vertices (first mesh query)")
        # PolyData.points are the mesh vertices — no second disk read via open3d needed.
        vertices = np.asarray(self._mesh_polydata.points)
        vf = features2vertex(vertices, self._result.points, self._lifted_normed)
        norms = np.linalg.norm(vf, axis=1, keepdims=True)
        self._mesh_vertex_features = (vf / (norms + 1e-8)).astype(np.float32)
```

(remove the now-unused `import open3d as o3d` lazy import; `ensure_mesh_features` is only called from `score_query` in mesh mode, which now runs after `ensure_mesh_polydata` — see step 7.)

6. In `score_query`'s mesh branch, guard the feature transfer behind the polydata:

```python
        if self.mode == "mesh":
            self.ensure_mesh_polydata(op_log=op_log)  # worker thread — disk read is safe here
            self.ensure_mesh_features(op_log)
```

7. Add a log line for the silent RGB fallback (after `self.ensure_lifted(op_log)`):

```python
        if self._lifted_normed is None:
            _stage("query: no semantic features for this scene — showing plain RGB")
            return self._result.colors
```

`app.py`:

8. Construct the viewer with the log: in `__init__` change `self._viewer = SplitViewer()` → `self._viewer = SplitViewer(op_log=self._op_log)`.

9. `_on_view_mode` — mesh mode now needs a worker round-trip for the disk read. Replace the method:

```python
    def _on_view_mode(self, event) -> None:
        """Switch pointcloud/mesh; load the mesh (worker) on first mesh view; keep query colours."""
        doc = pn.state.curdoc
        scene_key = self._current_scene

        def job():
            if event.new == "mesh":
                # First mesh view reads the ply — share the SceneCache with LocalizePage.
                preloaded = self._cache.get(scene_key, "mesh") if scene_key else None
                if self._viewer.ensure_mesh_polydata(preloaded=preloaded) and scene_key:
                    self._cache.put(scene_key, "mesh", self._viewer._mesh_polydata)
            query = self._viewer.active_query()
            if query and self._viewer.cached_query_colors(event.new) is None:
                positive, negative, extractor_name = query
                return self._viewer.score_query(
                    positive=positive, negative=negative, extractor_name=extractor_name, op_log=self._op_log
                )
            return None

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._viewer.set_mode(event.new)
            if res is not None:
                self._viewer.render_query(res)
            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op(f"switching to {event.new}")
        self._gpu.submit(job, on_done, doc)
```

(note: `set_mode` moves into `on_done` so rendering happens after the mesh is loaded; VTK mutation stays on the IOLoop.)

- [ ] **Step 4: Run to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py tests/dashboard/test_viewer_lift.py tests/dashboard/test_app.py -v`
Expected: PASS (adapt existing tests that asserted eager mesh reads or called `set_mode("mesh")` directly — they must call `ensure_mesh_polydata()` first, mirroring the app flow)

- [ ] **Step 5: Commit**

```bash
black collab_splats/dashboard/ tests/dashboard/ && isort collab_splats/dashboard/ tests/dashboard/
git add collab_splats/dashboard/viewer.py collab_splats/dashboard/app.py tests/dashboard/test_viewer.py tests/dashboard/test_app.py
git commit -m "perf(dashboard): lazy mesh load on worker; drop o3d re-read; surface viewer status"
```

---

### Task 12: Full-suite verification + docs

**Files:**
- Modify: `CLAUDE.md` (In-Flight Work), `docs/superpowers/plans/2026-07-16-dashboard-ux-feedback.md` (checkboxes)

- [ ] **Step 1: Full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -x -q`
Expected: green modulo `docs/known-test-failures.md`. Fix regressions before proceeding.

- [ ] **Step 2: Format sweep**

Run: `black . && isort .` — commit any formatting drift as `style(dashboard): black/isort pass`.

- [ ] **Step 3: Manual browser smoke checklist** (extends the owed dashboard checklist — record results in CLAUDE.md when done):

1. First click on Localize tab → spinner + "Building Localize page…" paints before the page appears; build time in log.
2. Start a scene load on Splats → Localize tab's Run + selectors grey out within ~300 ms; "busy: …" note shows current op.
3. Cold remote scene: click Run → page stays responsive; log shows "checking server for <stem>…", then pull % + "reading feedforward.zarr… done (Xs)".
4. Localize: select query video, drag frame slider → frame updates ~0.3 s after release; "frame N loaded (Xs)" in log; dragging fast shows only the final frame.
5. Localization run → figures appear without a page freeze; log shows DB/matching/figures step timings.
6. Switch view_mode to mesh on a fresh scene → "switching to mesh" + "reading mesh… done (Xs)" in log; widgets locked during the read.
7. Kill rclone (or disconnect) → dropdown populations log FAILED lines instead of sitting empty.

- [ ] **Step 4: Update CLAUDE.md + memory, final commit**

Move this work to "Recently completed" in CLAUDE.md (keep the manual-smoke-pending note if step 3 wasn't run in a browser). Commit:

```bash
git add -f CLAUDE.md docs/superpowers/plans/2026-07-16-dashboard-ux-feedback.md
git commit -m "docs: mark dashboard-ux-feedback plan complete"
```

---

## Self-review notes

- Spec §1 → Task 5. §2+§7 → Task 6 (poll-based deviation documented in header). §3 → Tasks 3, 7. §4 → Tasks 1, 9, 10. §5 → Tasks 5 (gc), 8 (has_processed, ffprobe), 10 (figures, localize mesh), 11 (viewer mesh). §6 → Tasks 2, 9, 11. §8 → Tasks 3 (seek), 4 (caches), 6 (poll idle), 7 (thumbnail), 8 (double fetch), 10+11 (shared mesh cache, lazy mesh). §8 frames.zarr double-write: deferred per spec.
- Types consistent: `step(label)` used identically everywhere; `set_busy(bool)` on LocalizePage vs `_set_busy(bool)` on SplatsApp (pre-existing naming kept); `ensure_mesh_polydata(preloaded, op_log) -> bool`.
- Tests referencing fixtures ("app_fixture", "page_fixture") must be adapted to each test file's existing construction pattern — the fixtures exist under other names; do not invent new heavyweight stubs.
