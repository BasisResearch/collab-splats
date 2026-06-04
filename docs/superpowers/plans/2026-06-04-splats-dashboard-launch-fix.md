# Splats Dashboard Launch Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the served dashboard from freezing the browser by moving all heavy/CUDA work off the asyncio IOLoop thread onto a single serialized GPU worker, with GPU memory reclaimed per job, off-loop session listing, configurable point decimation, and a tightened websocket origin.

**Architecture:** Producer/consumer. A process-singleton `GpuWorker` (one daemon thread + `queue.Queue`) runs every heavy job; IOLoop handlers only enqueue and toggle UI state. Results are marshalled back to the IOLoop via `doc.add_next_tick_callback`, where all VTK/panel mutation happens. The `doc` is captured at enqueue time (never `pn.state.curdoc` inside the worker).

**Tech Stack:** Python 3.11, Panel/Bokeh, PyVista/VTK, PyTorch (CUDA), pytest. Env: `/opt/venv/reconstruction/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-04-splats-dashboard-launch-fix-design.md`

---

## File Structure

- **Create** `collab_splats/dashboard/gpu_worker.py` — `GpuWorker`: serialized off-loop job runner, `pytorch_gc` per job, on-loop dispatch.
- **Create** `tests/dashboard/test_gpu_worker.py` — unit tests for the worker.
- **Modify** `collab_splats/dashboard/viewer.py` — split compute from render (`score_query` vs `_render_right`); add point decimation (`_decimate`, `max_points` on `load`).
- **Modify** `collab_splats/dashboard/app.py` — handlers enqueue jobs; `_set_busy` button gating; off-loop `_refresh_sessions`; `max_display_points` widget; `GpuWorker` injection.
- **Modify** `collab_splats/dashboard/__init__.py` — export `GpuWorker`.
- **Modify** `tests/dashboard/test_app.py` — update tests that assumed the old synchronous/threaded behaviour; add the regression test.
- **Modify** `tests/dashboard/test_viewer.py` — add decimation + split-query tests.

Run tests with: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -q`

---

## Task 1: GpuWorker — serialized off-loop job runner

**Files:**
- Create: `collab_splats/dashboard/gpu_worker.py`
- Test: `tests/dashboard/test_gpu_worker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dashboard/test_gpu_worker.py
"""Tests for the serialized GPU job worker."""
from unittest.mock import patch

from collab_splats.dashboard.gpu_worker import GpuWorker


class _FakeDoc:
    """Stand-in for a Bokeh document: runs the scheduled callback immediately."""
    def __init__(self):
        self.scheduled = []
    def add_next_tick_callback(self, cb):
        self.scheduled.append(cb)
        cb()  # emulate the IOLoop running it


def test_inline_when_doc_none_runs_job_and_on_done():
    w = GpuWorker()
    got = {}
    with patch("collab_splats.dashboard.gpu_worker.pytorch_gc") as gc:
        w.submit(job_fn=lambda: 21 * 2, on_done=lambda r: got.__setitem__("r", r), doc=None)
    assert got["r"] == 42
    gc.assert_called_once()  # cache cleared even on the inline path


def test_doc_path_dispatches_result_to_on_done():
    w = GpuWorker()
    doc = _FakeDoc()
    got = {}
    with patch("collab_splats.dashboard.gpu_worker.pytorch_gc"):
        w.submit(job_fn=lambda: "payload", on_done=lambda r: got.__setitem__("r", r), doc=doc)
        w.wait_idle(timeout=5)
    assert got["r"] == "payload"
    assert doc.scheduled  # render was marshalled to the IOLoop, not run on the worker


def test_job_exception_is_passed_to_on_done_and_worker_survives():
    w = GpuWorker()
    doc = _FakeDoc()
    results = []
    def boom():
        raise ValueError("kaboom")
    with patch("collab_splats.dashboard.gpu_worker.pytorch_gc") as gc:
        w.submit(job_fn=boom, on_done=lambda r: results.append(r), doc=doc)
        w.wait_idle(timeout=5)
        w.submit(job_fn=lambda: "ok", on_done=lambda r: results.append(r), doc=doc)
        w.wait_idle(timeout=5)
    assert isinstance(results[0], ValueError)   # error surfaced, not crashed
    assert results[1] == "ok"                   # worker survived to run the next job
    assert gc.call_count == 2                    # gc after every job (finally)


def test_pytorch_gc_called_after_each_job():
    w = GpuWorker()
    doc = _FakeDoc()
    with patch("collab_splats.dashboard.gpu_worker.pytorch_gc") as gc:
        w.submit(job_fn=lambda: 1, on_done=lambda r: None, doc=doc)
        w.wait_idle(timeout=5)
    gc.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_gpu_worker.py -q`
Expected: FAIL — `ModuleNotFoundError: collab_splats.dashboard.gpu_worker`

- [ ] **Step 3: Write the implementation**

```python
# collab_splats/dashboard/gpu_worker.py
"""Single serialized worker that runs all heavy/CUDA dashboard jobs off the IOLoop.

Panel/Bokeh serve on a single asyncio IOLoop thread. Heavy imports (vggt pulls a
module-level torch.compile -> inductor compile-worker pool) and CUDA inference must
never run on that thread: they block the websocket long enough for the bokeh session
token to expire, and the page never renders.

This worker owns ALL such work on one daemon thread, so jobs are serialized (no parallel
model loads -> no GPU/RAM OOM, no shared-state races) and torch.compile warms once. The
result is marshalled back to the IOLoop via doc.add_next_tick_callback, where VTK/panel
mutation is safe. The doc is captured by the caller at enqueue time — never pn.state.curdoc
inside the worker (it is thread-local and None off the IOLoop).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

from collab_splats.utils.torch_utils import pytorch_gc

logger = logging.getLogger(__name__)


class GpuWorker:
    """One daemon thread draining a job queue; pytorch_gc after every job."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self.busy = False
        self._thread = threading.Thread(target=self._loop, name="gpu-worker", daemon=True)
        self._thread.start()

    def submit(self, job_fn: Callable[[], Any], on_done: Callable[[Any], None], doc: Any) -> None:
        """Enqueue job_fn (runs on the worker); on_done(result_or_exception) runs on the IOLoop.

        doc is None (tests / non-served) -> run inline, synchronously, for testability.
        """
        if doc is None:
            result = self._run(job_fn)
            on_done(result)
            return
        self.busy = True
        self._queue.put((job_fn, on_done, doc))

    @staticmethod
    def _run(job_fn: Callable[[], Any]) -> Any:
        """Run a job, returning its result or the raised exception; always gc."""
        try:
            return job_fn()
        except Exception as exc:  # surface, never crash the worker
            logger.exception("gpu job failed")
            return exc
        finally:
            pytorch_gc()

    def _loop(self) -> None:
        while True:
            job_fn, on_done, doc = self._queue.get()
            result = self._run(job_fn)
            # Marshal the result back to the IOLoop; render + busy reset happen there.
            doc.add_next_tick_callback(lambda r=result, cb=on_done: self._finish(cb, r))

    def _finish(self, on_done: Callable[[Any], None], result: Any) -> None:
        """Runs on the IOLoop: clear busy then deliver the result to the handler."""
        if self._queue.empty():
            self.busy = False
        on_done(result)

    def wait_idle(self, timeout: float = 5.0) -> None:
        """Test helper: block until the queue is drained (best-effort)."""
        self._queue.join()
```

Note: add `self._queue.task_done()` so `wait_idle` works. Update `_loop`:

```python
    def _loop(self) -> None:
        while True:
            job_fn, on_done, doc = self._queue.get()
            try:
                result = self._run(job_fn)
                doc.add_next_tick_callback(lambda r=result, cb=on_done: self._finish(cb, r))
            finally:
                self._queue.task_done()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_gpu_worker.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Export GpuWorker**

Modify `collab_splats/dashboard/__init__.py`:

```python
# collab_splats/dashboard/__init__.py
"""Splats dashboard package."""

from collab_splats.dashboard.app import SplatsApp, run_app
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.operation_log import OperationLog

__all__ = ["SplatsApp", "run_app", "OperationLog", "GpuWorker"]
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/gpu_worker.py collab_splats/dashboard/__init__.py tests/dashboard/test_gpu_worker.py
git commit -m "feat(dashboard): serialized GpuWorker for off-loop CUDA jobs"
```

---

## Task 2: Viewer point decimation

**Files:**
- Modify: `collab_splats/dashboard/viewer.py`
- Test: `tests/dashboard/test_viewer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/dashboard/test_viewer.py
import numpy as np
from collab_splats.dashboard.viewer import _decimate_indices


def test_decimate_indices_caps_to_budget():
    idx = _decimate_indices(n=1000, max_points=150)
    assert idx.shape[0] == 150
    assert idx.max() < 1000
    assert len(np.unique(idx)) == 150  # no duplicates


def test_decimate_indices_noop_when_under_budget():
    idx = _decimate_indices(n=100, max_points=150)
    assert idx.shape[0] == 100
    assert np.array_equal(idx, np.arange(100))


def test_decimate_indices_nonpositive_budget_is_noop():
    idx = _decimate_indices(n=100, max_points=0)
    assert np.array_equal(idx, np.arange(100))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -q -k decimate`
Expected: FAIL — `cannot import name '_decimate_indices'`

- [ ] **Step 3: Add the helper and wire it into the panes**

In `collab_splats/dashboard/viewer.py`, add after the module-level helpers (after `load_lifted_normed`):

```python
def _decimate_indices(n: int, max_points: int) -> np.ndarray:
    """Return display indices into n points, evenly subsampled to at most max_points.

    Evenly-strided (deterministic, no RNG) so RGB and heatmap panes share the same
    subsample and stay registered. max_points <= 0 or n <= max_points -> identity.
    """
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, num=max_points, dtype=np.int64)
```

Change `SplitViewer.load` to accept and store a display budget + indices:

```python
    def load(self, result, mesh_path: Path | None, lifted_normed: np.ndarray | None = None,
             max_points: int = 150_000) -> None:
        """Load a FeedforwardResult (+ optional mesh + lifted features) into both panes."""
        self._result = result
        self._mesh_path = Path(mesh_path) if mesh_path else None
        self._lifted_normed = lifted_normed
        self._display_idx = _decimate_indices(len(result.points), max_points)
        self._render_left()
        self._render_right(None)
```

Initialise `self._display_idx = None` in `__init__` (next to `self._lifted_normed = None`).

Update `_render_left` pointcloud branch to decimate:

```python
        else:
            if self.mode == "mesh":
                self._status = "mesh.ply not found."
                logger.warning("mesh.ply not found; falling back to pointcloud for left pane")
            idx = self._display_idx
            cloud = pointcloud_to_polydata(self._result.points[idx], RGB=self._result.colors[idx])
            self.left_actor = self._left.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2)
```

Update `_render_right` to decimate both RGB-default and query-colour cases:

```python
    def _render_right(self, colors: np.ndarray | None) -> None:
        """Render RGB or similarity-colored pointcloud into the right plotter (decimated)."""
        self._right.clear()
        idx = self._display_idx
        rgb = colors if colors is not None else self._result.colors
        cloud = pointcloud_to_polydata(self._result.points[idx], RGB=rgb[idx])
        self.right_actor = self._right.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2)
        if not self._off_screen:
            self._right_pane.synchronize()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -q -k decimate`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/viewer.py tests/dashboard/test_viewer.py
git commit -m "feat(dashboard): decimate pointcloud panes to a display budget"
```

---

## Task 3: Split viewer query into off-loop compute + on-loop render

**Files:**
- Modify: `collab_splats/dashboard/viewer.py`
- Test: `tests/dashboard/test_viewer.py`

The current `SplitViewer.query` does CUDA scoring AND `_render_right` (synchronize) in one
call. Scoring must run off-loop; rendering must run on-loop. Split into `score_query`
(returns colours) and reuse `_render_right(colours)` on the IOLoop.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/dashboard/test_viewer.py
from unittest.mock import MagicMock
import torch
from collab_splats.dashboard.viewer import SplitViewer


def test_score_query_returns_colors_without_rendering():
    v = SplitViewer(off_screen=True)
    v._result = MagicMock()
    v._result.colors = np.zeros((4, 3), dtype=np.uint8)
    v._lifted_normed = np.eye(4, dtype=np.float32)
    fake = MagicMock()
    fake.score_queries.return_value = torch.tensor([0.1, 0.9, 0.5, 0.2])
    v._extractor_cache["talk2dino"] = fake
    colors = v.score_query(positive=["chair"], negative=["floor"], extractor_name="talk2dino")
    assert colors.shape == (4, 3)
    fake.score_queries.assert_called_once()


def test_score_query_blank_positive_returns_rgb():
    v = SplitViewer(off_screen=True)
    v._result = MagicMock()
    v._result.colors = np.full((4, 3), 7, dtype=np.uint8)
    v._lifted_normed = None
    colors = v.score_query(positive=[], negative=[], extractor_name="talk2dino")
    assert np.array_equal(colors, v._result.colors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -q -k score_query`
Expected: FAIL — `'SplitViewer' object has no attribute 'score_query'`

- [ ] **Step 3: Replace `query` with `score_query` (compute only)**

In `collab_splats/dashboard/viewer.py`, replace the `query` method with:

```python
    def score_query(
        self,
        positive: list[str],
        negative: list[str] | None = None,
        extractor_name: str = "talk2dino",
        op_log=None,
    ) -> np.ndarray:
        """Compute per-point query colours (RGB uint8). Pure compute — no rendering.

        Reuses BaseQueryableExtractor.score_queries (contrastive softmax, [0, 1]).
        Empty positive or no cached features -> returns the plain RGB colours.
        Call from the GPU worker; pass the returned colours to _render_right on the IOLoop.
        """
        def _stage(msg: str) -> None:
            if op_log is not None:
                op_log.append_line(msg)

        if not positive or self._lifted_normed is None:
            return self._result.colors

        _stage(f"query: encoding {len(positive)} positive / {len(negative or [])} negative")
        extractor = self._get_extractor(extractor_name)
        features = torch.from_numpy(self._lifted_normed)  # (P, D)

        _stage(f"query: scoring {features.shape[0]} points")
        scores = extractor.score_queries(features, positive=positive, negative=negative or None)
        sims = scores.detach().cpu().numpy()
        colors = apply_viridis(sims)
        _stage("query: scored")
        return colors

    def render_query(self, colors: np.ndarray) -> None:
        """Recolour the right pane with precomputed query colours (IOLoop thread)."""
        self._render_right(colors)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -q -k score_query`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/viewer.py tests/dashboard/test_viewer.py
git commit -m "refactor(dashboard): split viewer query into score (off-loop) + render (on-loop)"
```

---

## Task 4: App — busy-state button gating + GpuWorker injection

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/dashboard/test_app.py
from collab_splats.dashboard.gpu_worker import GpuWorker


def test_set_busy_toggles_action_buttons(tmp_path):
    app, _ = _app(tmp_path)
    app._set_busy(True)
    assert app.run_btn.disabled and app.force_btn.disabled and app.run_query_btn.disabled
    app._set_busy(False)
    assert not app.run_btn.disabled and not app.force_btn.disabled and not app.run_query_btn.disabled


def test_has_max_display_points_widget(tmp_path):
    app, _ = _app(tmp_path)
    assert app.max_display_points.value == 150_000


def test_app_uses_injected_gpu_worker(tmp_path):
    from unittest.mock import MagicMock, patch
    worker = MagicMock(spec=GpuWorker)
    source = MagicMock()
    source.list_sessions.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker)
    assert app._gpu is worker
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k "set_busy or injected_gpu"`
Expected: FAIL — `SplatsApp.__init__() got an unexpected keyword argument 'gpu_worker'` / no `_set_busy`

- [ ] **Step 3: Add the GpuWorker param, `_set_busy`, and default construction**

In `collab_splats/dashboard/app.py`, update imports near the top:

```python
from collab_splats.dashboard.gpu_worker import GpuWorker
```

Update `SplatsApp.__init__` signature and body:

```python
    def __init__(
        self,
        base_dir: Path = Path("/workspace/outputs"),
        source: SessionSource | None = None,
        gpu_worker: GpuWorker | None = None,
        **params,
    ) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._source = source if source is not None else SessionSource()
        self._gpu = gpu_worker if gpu_worker is not None else GpuWorker()
        self._op_log = OperationLog()
        self._viewer = SplitViewer()
        self._build_sidebar()
        self._refresh_sessions()
```

Add `_set_busy` (after `_build_sidebar`):

```python
    def _set_busy(self, busy: bool) -> None:
        """Enable/disable the action buttons while a GPU job is in flight (IOLoop thread)."""
        for btn in (self.run_btn, self.force_btn, self.run_query_btn):
            btn.disabled = busy
```

Add the `max_display_points` widget in `_build_sidebar` (next to the mesh widgets, before
the watcher wiring):

```python
        self.max_display_points = pn.widgets.IntInput(name="Max display points", value=150_000, step=50_000)
```

And place it in `self._sidebar` just before `"### View"`:

```python
            self.max_display_points,
            "### View",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k "set_busy or injected_gpu or max_display"`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): inject GpuWorker, add busy-state button gating"
```

---

## Task 5: App — `_load_outputs` enqueues an off-loop job (the core fix + regression test)

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing regression test**

```python
# append to tests/dashboard/test_app.py
class _RecordingWorker:
    """Captures submitted jobs WITHOUT running them — proves work is deferred off-loop."""
    def __init__(self):
        self.submitted = []
    def submit(self, job_fn, on_done, doc):
        self.submitted.append((job_fn, on_done, doc))


def test_load_outputs_defers_heavy_work_to_worker(tmp_path):
    from unittest.mock import MagicMock, patch
    worker = _RecordingWorker()
    source = MagicMock()
    source.list_sessions.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker)
    out = tmp_path / "s" / "clip" / "feedforward.zarr"
    out.mkdir(parents=True)
    app._load_outputs("s", "clip")
    # The handler must NOT render inline; it enqueues exactly one job.
    app._viewer.load.assert_not_called()
    assert len(worker.submitted) == 1
    assert callable(worker.submitted[0][0])  # job_fn deferred to the worker


def test_load_outputs_on_done_renders_into_viewer(tmp_path):
    from unittest.mock import MagicMock, patch
    worker = _RecordingWorker()
    source = MagicMock()
    source.list_sessions.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker)
    (tmp_path / "s" / "clip" / "feedforward.zarr").mkdir(parents=True)
    app._load_outputs("s", "clip")
    _job, on_done, _doc = worker.submitted[0]
    sentinel = ("result", None, "lifted")
    on_done(sentinel)
    app._viewer.load.assert_called_once()
    kwargs = app._viewer.load.call_args.kwargs
    assert kwargs["max_points"] == app.max_display_points.value
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k load_outputs`
Expected: FAIL — `_load_outputs` still imports + renders inline (`viewer.load` called, no `submitted`)

- [ ] **Step 3: Rewrite `_load_outputs` to enqueue a job**

Replace `_load_outputs` and `_dispatch_load` in `collab_splats/dashboard/app.py` with:

```python
    def _dispatch_load(self, session: str, stem: str) -> None:
        """Schedule an outputs load (called from a worker job's completion)."""
        self._load_outputs(session, stem)

    def _load_outputs(self, session: str, stem: str) -> None:
        """Enqueue loading FeedforwardResult + semantics; render on the IOLoop when done."""
        out = self._base_dir / session / stem
        doc = pn.state.curdoc  # captured on the IOLoop at call time
        max_points = self.max_display_points.value

        def job():
            # Lazy import: FeedforwardResult lives in the heavy feedforward package.
            from collab_splats.pointcloud.feedforward.base import FeedforwardResult

            if not (out / "feedforward.zarr").exists():
                self._source.pull_processed(session, stem, out)
            result = FeedforwardResult.load_zarr(out / "feedforward.zarr")
            try:
                lifted = load_lifted_normed(result, out / "semantics")
            except Exception:
                lifted = None
            mesh_path = out / "mesh" / "mesh.ply"
            return (result, mesh_path if mesh_path.exists() else None, lifted)

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            result, mesh_path, lifted = res
            self._viewer.load(result, mesh_path=mesh_path, lifted_normed=lifted, max_points=max_points)
            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op(f"loading {stem}")
        self._gpu.submit(job, on_done, doc)
```

Note: `_dispatch_load` is retained as a thin alias because `_on_run`'s worker calls it
(updated in Task 6). Remove the old `from ...FeedforwardResult` lazy import line that lived
in the previous `_load_outputs` body.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k load_outputs`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "fix(dashboard): load outputs off the IOLoop via GpuWorker (core freeze fix)"
```

---

## Task 6: App — `_on_run` enqueues the pipeline job

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Update the existing run tests for the worker API**

Replace `test_run_button_spawns_pipeline` and `test_force_rerun_recomputes_even_when_cached`
in `tests/dashboard/test_app.py` with worker-based versions:

```python
def test_run_button_submits_pipeline_job(tmp_path):
    from unittest.mock import MagicMock, patch
    worker = _RecordingWorker()
    source = MagicMock()
    source.list_sessions.return_value = ["2026_05_07"]
    source.list_videos.return_value = ["clip_03.mp4"]
    source.has_processed.return_value = False
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    app._on_run(event=None, force=True)
    assert len(worker.submitted) == 1  # pipeline deferred to the worker
    assert app.run_btn.disabled         # busy while running


def test_force_rerun_submits_even_when_cached(tmp_path):
    from unittest.mock import MagicMock, patch
    worker = _RecordingWorker()
    source = MagicMock()
    source.list_sessions.return_value = ["2026_05_07"]
    source.list_videos.return_value = ["clip_03.mp4"]
    source.has_processed.return_value = False
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    (tmp_path / "2026_05_07" / "clip_03" / "feedforward.zarr").mkdir(parents=True)
    app._on_run(event=None, force=True)
    assert len(worker.submitted) == 1  # recompute despite cache
```

Also update `test_run_loads_cache_without_recompute`: it patches `_load_outputs`, which is
still called on the cached path — keep it but drop the `threading.Thread` patch:

```python
def test_run_loads_cache_without_recompute(tmp_path):
    from unittest.mock import patch
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    out = tmp_path / "2026_05_07" / "clip_03" / "feedforward.zarr"
    out.mkdir(parents=True)
    with patch.object(app, "_load_outputs") as load:
        app._on_run(event=None, force=False)
    load.assert_called_once()  # loaded from cache, no recompute job
```

Update `_app` helper to give the default app a recording worker so its handlers don't spawn
a real `GpuWorker` thread:

```python
def _app(tmp_path):
    source = MagicMock()
    source.list_sessions.return_value = ["2026_05_07"]
    source.list_videos.return_value = ["clip_03.mp4"]
    source.has_processed.return_value = False
    with patch("collab_splats.dashboard.app.SplitViewer"):
        return SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker()), source
```

(Move the `_RecordingWorker` class definition above `_app` in the file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k "run_button or force_rerun or loads_cache"`
Expected: FAIL — `_on_run` still uses `threading.Thread`, no `worker.submitted`

- [ ] **Step 3: Rewrite `_on_run` to enqueue the pipeline job**

Replace `_on_run` in `collab_splats/dashboard/app.py` with:

```python
    def _on_run(self, event, force: bool) -> None:
        """Run or reload the pipeline off the IOLoop, respecting cache and force flag."""
        session, name = self.session_select.value, self.video_select.value
        if not session or not name:
            return
        stem = Path(name).stem
        out = self._base_dir / session / stem
        cached = (out / "feedforward.zarr").exists() or self._source.has_processed(session, stem)
        # Cached and not forced: just load existing outputs (itself an off-loop job).
        if cached and not force:
            self._load_outputs(session, stem)
            return
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
            self._load_outputs(session, stem)  # re-enqueues a load job

        self._set_busy(True)
        self._op_log.start_op(f"running {stem}")
        self._gpu.submit(job, on_done, doc)
```

Remove the now-unused `threading` import only if nothing else uses it (the `_ensure_display`
Xvfb `subprocess` block does not need `threading`; check and leave `import threading` if any
remaining reference exists — `_refresh_sessions` in Task 8 will use it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k "run_button or force_rerun or loads_cache"`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "fix(dashboard): run pipeline off the IOLoop via GpuWorker"
```

---

## Task 7: App — `_on_query` enqueues score job, renders on-loop

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Replace the old query test**

Replace `test_run_query_button_forwards_parsed_terms` in `tests/dashboard/test_app.py`:

```python
def test_query_submits_score_job_with_parsed_terms(tmp_path):
    app, _ = _app(tmp_path)
    app.pos_query.value = "chair, stool"
    app.neg_query.value = "floor"
    app._on_query(event=None)
    # Deferred to the worker, not scored inline.
    assert len(app._gpu.submitted) == 1
    assert app.run_query_btn.disabled


def test_query_on_done_renders_colors(tmp_path):
    import numpy as np
    app, _ = _app(tmp_path)
    app.pos_query.value = "chair"
    app._on_query(event=None)
    _job, on_done, _doc = app._gpu.submitted[0]
    colors = np.zeros((3, 3), dtype=np.uint8)
    on_done(colors)
    app._viewer.render_query.assert_called_once_with(colors)
    assert not app.run_query_btn.disabled  # re-enabled after render
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k query`
Expected: FAIL — `_on_query` still calls `_viewer.query` inline

- [ ] **Step 3: Rewrite `_on_query`**

Replace `_on_query` in `collab_splats/dashboard/app.py` with:

```python
    def _on_query(self, event) -> None:
        """Score the positive/negative query off the IOLoop; recolour the right pane on done."""
        positive = _split_terms(self.pos_query.value)
        negative = _split_terms(self.neg_query.value)
        extractor_name = self.extractor.value
        doc = pn.state.curdoc

        def job():
            return self._viewer.score_query(
                positive=positive, negative=negative, extractor_name=extractor_name, op_log=self._op_log
            )

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._viewer.render_query(res)

        self._set_busy(True)
        self._op_log.start_op("query")
        self._gpu.submit(job, on_done, doc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k query`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "fix(dashboard): score queries off the IOLoop, render on-loop"
```

---

## Task 8: App — off-loop session listing

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/dashboard/test_app.py
def test_refresh_sessions_runs_off_loop(tmp_path):
    from unittest.mock import MagicMock, patch
    source = MagicMock()
    source.list_sessions.return_value = ["a", "b"]
    with patch("collab_splats.dashboard.app.SplitViewer"), \
         patch("collab_splats.dashboard.app.threading.Thread") as thread:
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    thread.assert_called()  # listing dispatched to a background thread, not inline on the loop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k off_loop`
Expected: FAIL — `_refresh_sessions` calls `list_sessions` inline (no Thread)

- [ ] **Step 3: Make `_refresh_sessions` off-loop**

Rewrite `_refresh_sessions` to list off the IOLoop and dispatch options back:

```python
    def _refresh_sessions(self) -> None:
        """List sessions on a background thread; set options back on the IOLoop."""
        doc = pn.state.curdoc

        def work():
            try:
                names = self._source.list_sessions()
            except Exception as exc:
                logger.warning("session listing failed: %s", exc)
                names = []
            self._apply_sessions(names, doc)

        threading.Thread(target=work, name="session-list", daemon=True).start()

    def _apply_sessions(self, names: list[str], doc) -> None:
        """Set the session dropdown options on the IOLoop (or inline if no doc)."""
        def setter():
            self.session_select.options = names
        if doc is not None:
            doc.add_next_tick_callback(setter)
        else:
            setter()
```

Keep `import threading` at the top of `app.py` (now used here).

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k off_loop`
Expected: PASS. Note: existing `test_app_populates_sessions` now needs the inline
(doc=None) path — `_apply_sessions` with `doc=None` sets options synchronously, so it still
passes.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): off-loop session listing + max-display-points control"
```

---

## Task 9: `run_app` — GpuWorker singleton, token expiry, websocket origin

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/dashboard/test_app.py
def test_run_app_serves_with_hardening(tmp_path):
    from unittest.mock import patch
    with patch("collab_splats.dashboard.app._ensure_display"), \
         patch("collab_splats.dashboard.app.pn.extension"), \
         patch("collab_splats.dashboard.app.GpuWorker") as worker_cls, \
         patch("collab_splats.dashboard.app.pn.serve") as serve:
        from collab_splats.dashboard.app import run_app
        run_app(host="127.0.0.1", port=9999, base_dir=str(tmp_path), websocket_origin=None)
    worker_cls.assert_called_once()  # one shared worker for all sessions
    kwargs = serve.call_args.kwargs
    assert kwargs["session_token_expiration"] >= 1800
    assert kwargs["websocket_origin"] == ["127.0.0.1:9999", "localhost:9999"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k run_app_serves`
Expected: FAIL — no `session_token_expiration`; `websocket_origin` default is `"*"`; no shared worker

- [ ] **Step 3: Update `run_app`**

Replace `run_app` in `collab_splats/dashboard/app.py` with:

```python
def run_app(
    host: str = "0.0.0.0",
    port: int = 7860,
    base_dir: str = "/workspace/outputs",
    websocket_origin: str | list[str] | None = None,
) -> None:
    """Serve the splats dashboard.

    websocket_origin=None restricts connections to host:port + localhost:port. Pass an
    explicit list (or "*") to allow remote-IP / SSH-tunnel access.
    """
    # Headless host: ensure an OpenGL context exists before any VTK initialisation.
    _ensure_display()

    # Load the VTK extension ONCE here, in the main thread, before serving. inline=True
    # serves all JS/CSS from this server (no CDN) for headless/air-gapped hosts.
    pn.extension("vtk", inline=True)

    # One shared GPU worker for every session: serializes all CUDA work across tabs,
    # preventing parallel model loads from OOMing the GPU.
    gpu_worker = GpuWorker()

    if websocket_origin is None:
        origin: str | list[str] = [f"{host}:{port}", f"localhost:{port}"]
    else:
        origin = websocket_origin

    def factory() -> pn.template.MaterialTemplate:
        return SplatsApp(base_dir=Path(base_dir), gpu_worker=gpu_worker).view()

    pn.serve(
        factory,
        address=host,
        port=port,
        show=False,
        title="splats",
        websocket_origin=origin,
        session_token_expiration=1800,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q -k run_app_serves`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): shared GpuWorker, raise token expiry, scope websocket origin"
```

---

## Task 10: Full suite green + format

**Files:** none (verification)

- [ ] **Step 1: Run the full dashboard + semantics suites**

Run:
```bash
/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -q
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/ -q
```
Expected: all green. If `test_app_populates_sessions` / `test_selecting_session_lists_videos`
fail because options are now set via `_apply_sessions(doc=None)` inline, confirm the inline
path runs synchronously; fix any test that assumed the old direct assignment.

- [ ] **Step 2: Format**

Run: `black collab_splats/dashboard tests/dashboard && isort collab_splats/dashboard tests/dashboard`

- [ ] **Step 3: Commit any formatting**

```bash
git add -A && git commit -m "style(dashboard): black + isort" || echo "nothing to format"
```

---

## Task 11: Manual smoke-test (real browser)

**Files:** none (manual verification — not automated; see spec "Open risk")

- [ ] **Step 1: Clear any stale instance**

```bash
pkill -9 -f collab_splats.dashboard; pkill -9 Xvfb
ps -eo pid,cmd | grep collab_splats.dashboard | grep -v grep   # expect nothing
```

- [ ] **Step 2: Launch in tmux**

```bash
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --port 7860
```
Expected: `Launching server at http://0.0.0.0:7860`, rclone verify, no Warp banner at launch
(heavy stack warms on first Run/query, not at boot).

- [ ] **Step 3: Verify in browser**

1. Page renders two viewer panes within a few seconds — **no** "Token is expired", no frozen tab.
2. Select a cached session/video → outputs load; buttons disable while loading, re-enable after; pane shows the (decimated) pointcloud.
3. Positive/negative query + Run query → right pane recolours; buttons disable during scoring.
4. Lower **Max display points** → reload → panes render fewer points, more responsive.
5. Confirm only one job runs at a time (buttons disabled while busy); no thread explosion (`ps -o nlwp -p <pid>` stays bounded, not 400+).

- [ ] **Step 4: Record the result**

Append a short note to `docs/superpowers/handoffs/2026-06-04-splats-dashboard-launch-fix-handoff.md`
with pass/fail per smoke-test item and the observed thread count.
