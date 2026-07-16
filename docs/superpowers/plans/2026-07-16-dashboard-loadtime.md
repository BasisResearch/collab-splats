# Dashboard Load-Time & Display-Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut dashboard launch + scene-load + interaction latency, make rclone network activity visible with a live progress bar, and stop the GPU-worker crash — without changing the on-disk zarr format.

**Architecture:** Dashboard is a Panel/Bokeh app on one IOLoop with a single serialized `GpuWorker` daemon thread for CUDA/heavy work. Fixes: guard cross-thread callbacks, stop over-pulling from GCS, stream rclone stats to the shared `OperationLog`, push blocking rclone off the IOLoop, memoize listings, cache scene/mesh state in memory, and defer/skip redundant decode + render work.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), Panel/Bokeh, PyVista/VTK, zarr 3.x, rclone (via `collab_data` `RcloneClient`), pytest.

**Reference spec:** `docs/superpowers/specs/2026-07-16-dashboard-loadtime-design.md`

**Run all tests with:** `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -v`

---

## File Structure

- `collab_splats/dashboard/gpu_worker.py` — Task 1 (dead-doc guard)
- `collab_splats/dashboard/sources.py` — Tasks 2 (`_PULL_EXCLUDES` home + excludes on pull), 3 (rclone `--stats` streaming + percent parse), 6 (listing memoization)
- `collab_splats/dashboard/pipeline.py` — Task 2 (import `_PULL_EXCLUDES` from shared home)
- `collab_splats/dashboard/app.py` — Tasks 2 (splats pull uses excludes), 4 (progress wiring + fetch start/finish), 5 (rclone off IOLoop), 9 (SceneCache), 10 (warm stack), 13 (defer lifted), 15 (debounce state)
- `collab_splats/dashboard/localize.py` — Task 5 (rclone watchers off IOLoop)
- `collab_splats/dashboard/viewer.py` — Tasks 7 (mesh cache + non-mutating normalize), 8 (recolor without rebuild), 12 (skip per-render overhead)
- `collab_splats/dashboard/viz_utils.py` — Task 11 (cheaper points)
- `collab_splats/dashboard/shell.py` — Tasks 9 (pass cache to SplatsApp), 16 (lazy plotter)
- `collab_splats/pointcloud/feedforward/base.py` — Task 14 (opt-in dense decode)
- Tests mirror under `tests/dashboard/` and `tests/pointcloud/feedforward/`.

---

# PHASE 1 — Core (high impact, low risk)

## Task 1: GPU worker survives a dead document (F5)

**Files:**
- Modify: `collab_splats/dashboard/gpu_worker.py:59-73`
- Test: `tests/dashboard/test_gpu_worker.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_gpu_worker.py`:

```python
def test_worker_survives_dead_document():
    """A destroyed session raises in add_next_tick_callback; the worker must not die."""
    import threading

    w = GpuWorker()

    class _DeadDoc:
        def add_next_tick_callback(self, cb):
            raise AttributeError(
                "'DocumentCallbackManager' object has no attribute '_change_callbacks'"
            )

    done = threading.Event()

    # First job targets a dead doc -> marshal raises inside the worker loop.
    w.submit(lambda: 1, lambda r: None, _DeadDoc())

    # Second job targets a live doc; it must still run -> proves the worker thread lived.
    class _LiveDoc:
        def add_next_tick_callback(self, cb):
            cb()

    w.submit(lambda: 2, lambda r: done.set(), _LiveDoc())
    assert done.wait(timeout=5.0), "worker thread died after a dead-document marshal"
    assert w.busy is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_gpu_worker.py::test_worker_survives_dead_document -v`
Expected: FAIL — the worker thread dies on the `AttributeError`, second job never runs, `done.wait` times out.

- [ ] **Step 3: Guard the marshal in `_loop`**

In `collab_splats/dashboard/gpu_worker.py`, replace `_loop` (lines 59-67):

```python
    def _loop(self) -> None:
        while True:
            job_fn, on_done, doc = self._queue.get()
            try:
                result = self._run(job_fn)
                # Marshal the result back to the IOLoop; render + busy reset happen there.
                try:
                    doc.add_next_tick_callback(lambda r=result, cb=on_done: self._finish(cb, r))
                except Exception:
                    # Session gone (token expired / tab closed): the doc's callback manager is
                    # torn down and add_next_tick_callback raises. Drop the result and keep the
                    # worker alive so a reconnecting session still gets a working dashboard.
                    logger.debug("dropping result for a destroyed session", exc_info=True)
                    self.busy = False
            finally:
                self._queue.task_done()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_gpu_worker.py -v`
Expected: PASS (all worker tests).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/gpu_worker.py tests/dashboard/test_gpu_worker.py
git commit -m "fix(dashboard): GPU worker survives dead-document marshal (F5)"
```

---

## Task 2: Splats pull excludes dense arrays (F6)

**Files:**
- Modify: `collab_splats/dashboard/sources.py` (add module-level `PULL_EXCLUDES`)
- Modify: `collab_splats/dashboard/pipeline.py` (import shared constant instead of local `_PULL_EXCLUDES`)
- Modify: `collab_splats/dashboard/app.py:429-449` (`_load_outputs` job passes excludes)
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_app.py`:

```python
def test_load_outputs_pull_excludes_dense_arrays(tmp_path):
    """The splats load must skip GBs of dense arrays the viewer never reads."""
    import numpy as np

    from collab_splats.dashboard.sources import PULL_EXCLUDES

    app = _recording_app(tmp_path)  # existing helper: SplatsApp + _RecordingWorker
    app.session_select.value = "2026_05_07"
    app.video_select.options = ["clip_03.mp4"]

    app._load_outputs("2026_05_07", "clip_03")
    job_fn, _on_done, _doc = app._gpu.submitted[-1]

    app._source.pull_processed.reset_mock()
    # feedforward.zarr absent -> job pulls; assert it forwards the exclude set.
    try:
        job_fn()
    except Exception:
        pass  # load_zarr will fail on the empty tmp tree; we only assert the pull call
    _args, kwargs = app._source.pull_processed.call_args
    assert kwargs.get("excludes") == PULL_EXCLUDES or (len(_args) >= 4 and _args[3] == PULL_EXCLUDES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py::test_load_outputs_pull_excludes_dense_arrays -v`
Expected: FAIL — `pull_processed` is called with no `excludes` (import of `PULL_EXCLUDES` also fails until Step 3).

- [ ] **Step 3: Add the shared constant in `sources.py`**

In `collab_splats/dashboard/sources.py`, under the `# Constants` divider (after line 24), add:

```python
# Members skipped on every processed-scene pull: frames.zarr duplicates the frames/ jpg dir,
# and the dense per-pixel arrays are optional in FeedforwardResult.load_zarr (absent -> None)
# and unused by both the splats viewer and localization — they can be GBs per scene.
PULL_EXCLUDES = (
    "frames.zarr/**",
    "feedforward.zarr/depth/**",
    "feedforward.zarr/world_points/**",
    "feedforward.zarr/confidence/**",
    "feedforward.zarr/conf/**",  # legacy key for confidence
    "feedforward.zarr/features/**",
    "feedforward.zarr/pixel_indices/**",
    "feedforward.zarr/images/**",
)
```

- [ ] **Step 4: Point `pipeline.py` at the shared constant**

In `collab_splats/dashboard/pipeline.py`, delete the local `_PULL_EXCLUDES` tuple (added by `0bf9ac4`) and import the shared one. Add to the imports block:

```python
from collab_splats.dashboard.sources import PULL_EXCLUDES, SessionSource
```

Replace every local `_PULL_EXCLUDES` reference in `pipeline.py` with `PULL_EXCLUDES`.

- [ ] **Step 5: Splats load passes excludes**

In `collab_splats/dashboard/app.py`, add to the imports block near line 23:

```python
from collab_splats.dashboard.sources import PULL_EXCLUDES
```

In `_load_outputs`'s `job()` (line 433-434), change the pull to forward the excludes:

```python
            if not (out / "feedforward.zarr").exists():
                self._source.pull_processed(session, stem, out, excludes=PULL_EXCLUDES)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/sources.py collab_splats/dashboard/pipeline.py collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): splats pull excludes dense arrays via shared PULL_EXCLUDES (F6)"
```

---

## Task 3: rclone `--stats` streaming + percent parse (F3, sources layer)

**Files:**
- Modify: `collab_splats/dashboard/sources.py` (`pull_processed`, `fetch_video` stream stats + `on_line`; add `parse_rclone_percent`)
- Test: `tests/dashboard/test_sources.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/dashboard/test_sources.py`:

```python
def test_parse_rclone_percent_extracts_percentage():
    from collab_splats.dashboard.sources import parse_rclone_percent

    line = "Transferred:   1.234 GiB / 5.678 GiB, 21%, 45.6 MiB/s, ETA 1m30s"
    assert parse_rclone_percent(line) == 21
    assert parse_rclone_percent("no percent here") is None
    assert parse_rclone_percent("Transferred: 0 / 0 Bytes, 100%, 0/s") == 100


def test_pull_processed_streams_stats_to_on_line(monkeypatch, tmp_path):
    lines_seen = []

    class _FakeProc:
        stdout = iter(["Transferred: 1 GiB / 2 GiB, 50%, 10 MiB/s\n"])

        def wait(self):
            return 0

    def fake_popen(cmd, stdout, stderr, text):
        assert "--stats-one-line" in cmd
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    src = SessionSource(client=_client())
    src.pull_processed("2026_05_07", "clip_03", tmp_path, on_line=lines_seen.append)
    assert any("50%" in ln for ln in lines_seen)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_sources.py::test_parse_rclone_percent_extracts_percentage tests/dashboard/test_sources.py::test_pull_processed_streams_stats_to_on_line -v`
Expected: FAIL — `parse_rclone_percent` undefined; `pull_processed` has no `on_line` param and uses `subprocess.run` (not `Popen` with `--stats`).

- [ ] **Step 3: Add the percent parser**

In `collab_splats/dashboard/sources.py`, add near the top (after the imports, add `import re` if not present — it already imports `re`):

```python
# rclone --stats line looks like: "Transferred: 1.2 GiB / 5.6 GiB, 21%, 45 MiB/s, ETA 1m"
_RCLONE_PCT_RE = re.compile(r",\s*(\d{1,3})%")


def parse_rclone_percent(line: str) -> "int | None":
    """Extract the integer transfer percentage from an rclone --stats line, or None."""
    m = _RCLONE_PCT_RE.search(line)
    if not m:
        return None
    return min(100, int(m.group(1)))
```

- [ ] **Step 4: Stream stats in `pull_processed`**

Replace `pull_processed` (lines 118-133) with an `on_line`-streaming version mirroring `push_outputs`:

```python
    def pull_processed(
        self, session: str, stem: str, dest_dir: Path, excludes: tuple = (), on_line=None
    ) -> Path:
        """rclone-copy processed outputs to dest_dir; return the local dir.

        excludes: rclone --exclude patterns (e.g. "frames.zarr/**") to skip artifacts a
        consumer does not need. on_line, if given, receives each --stats progress line.
        """
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{client.remote_name}:{PROCESSED_BUCKET}/{ROOT}/{session}/{stem}"
        flags: list[str] = []
        for pattern in excludes:
            flags += ["--exclude", pattern]
        flags += ["--stats", "2s", "--stats-one-line"]
        proc = subprocess.Popen(
            client._cmd("copy", *flags, remote, str(dest_dir)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout or []:
            line = line.strip()
            if line and on_line is not None:
                on_line(line)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"rclone copy failed (exit {ret}) for {remote}")
        return dest_dir
```

- [ ] **Step 5: Stream stats in `fetch_video`**

Replace `fetch_video` (lines 62-70) with a streaming version:

```python
    def fetch_video(self, session: str, name: str, dest_dir: Path, on_line=None) -> Path:
        """rclone-copy a remote video to dest_dir; return the local path. on_line gets --stats lines."""
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        remote = f"{client.remote_name}:{CURATED_BUCKET}/{ROOT}/{session}/{name}"
        proc = subprocess.Popen(
            client._cmd("copyto", "--stats", "2s", "--stats-one-line", remote, str(local)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout or []:
            line = line.strip()
            if line and on_line is not None:
                on_line(line)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"rclone copyto failed (exit {ret}) for {remote}")
        return local
```

- [ ] **Step 6: Update the existing `fetch_video`/`pull_processed` tests for Popen**

The pre-existing `test_fetch_video_invokes_rclone_copyto` and `test_pull_processed_invokes_rclone_copy` monkeypatch `subprocess.run`; both verbs now use `Popen`. Update each to patch `subprocess.Popen` returning a fake proc with `stdout=iter([])` and `wait()->0`, asserting the command contains the expected verb (`copyto` / `copy`) and remote. Example replacement for the pull test:

```python
def test_pull_processed_invokes_rclone_copy(monkeypatch, tmp_path):
    seen = {}

    class _FakeProc:
        stdout = iter([])

        def wait(self):
            return 0

    def fake_popen(cmd, stdout, stderr, text):
        seen["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    SessionSource(client=_client()).pull_processed("2026_05_07", "clip_03", tmp_path)
    assert "copy" in seen["cmd"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_sources.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/dashboard/sources.py tests/dashboard/test_sources.py
git commit -m "feat(dashboard): stream rclone --stats from pull/fetch + percent parser (F3)"
```

---

## Task 4: Wire pull/fetch progress to the status bar (F3, app layer)

**Files:**
- Modify: `collab_splats/dashboard/app.py` (`_load_outputs` job pipes `on_line`; `_ensure_local_video`/`_update_max_frames_bound` report to op_log)
- Modify: `collab_splats/dashboard/localize.py` (`_ensure_local_query_video` reports)
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_app.py`:

```python
def test_load_job_reports_pull_progress_to_op_log(tmp_path):
    """rclone --stats lines must drive op_log.update_progress so the bar shows a live %."""
    app = _recording_app(tmp_path)
    seen_pct = []
    app._op_log.update_progress = lambda pct, message="", log=True: seen_pct.append(pct)

    # pull_processed invokes on_line with a stats line carrying 42%.
    def fake_pull(session, stem, out, excludes=(), on_line=None):
        if on_line:
            on_line("Transferred: 1 GiB / 2 GiB, 42%, 10 MiB/s")
        (out / "feedforward.zarr").mkdir(parents=True, exist_ok=True)
        raise RuntimeError("stop before load_zarr")

    app._source.pull_processed = fake_pull
    app._load_outputs("2026_05_07", "clip_03")
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    try:
        job_fn()
    except Exception:
        pass
    assert 42 in seen_pct
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py::test_load_job_reports_pull_progress_to_op_log -v`
Expected: FAIL — the pull is called with no `on_line`, so `update_progress` never gets 42.

- [ ] **Step 3: Pipe pull progress in `_load_outputs`**

In `collab_splats/dashboard/app.py`, add to the imports block:

```python
from collab_splats.dashboard.sources import PULL_EXCLUDES, parse_rclone_percent
```

(Merge with the `PULL_EXCLUDES` import added in Task 2.) In `_load_outputs`'s `job()`, replace the pull block:

```python
            if not (out / "feedforward.zarr").exists():

                def _pull_progress(line: str) -> None:
                    pct = parse_rclone_percent(line)
                    if pct is not None:
                        self._op_log.update_progress(pct, "⬇ pulling from server", log=False)

                self._source.pull_processed(
                    session, stem, out, excludes=PULL_EXCLUDES, on_line=_pull_progress
                )
```

- [ ] **Step 4: Report video fetches to op_log**

In `_ensure_local_video` (lines 370-375), stream fetch progress + bracket with start/finish so a remote video fetch shows on the bar:

```python
    def _ensure_local_video(self, session: str, name: str) -> Path:
        """Return local video path, fetching from source if needed (progress -> op_log)."""
        local = self._base_dir / session / Path(name).stem / name
        if local.exists():
            return local

        def _progress(line: str) -> None:
            pct = parse_rclone_percent(line)
            if pct is not None:
                self._op_log.update_progress(pct, "⬇ fetching video", log=False)

        return self._source.fetch_video(session, name, local.parent, on_line=_progress)
```

- [ ] **Step 5: Report the localize query-video fetch**

In `collab_splats/dashboard/localize.py`, add `from collab_splats.dashboard.sources import parse_rclone_percent` to the imports, and in `_ensure_local_query_video` (lines 290-294) pass an `on_line` that calls `self._op_log.update_progress(pct, "⬇ fetching query video", log=False)`, mirroring Step 4.

- [ ] **Step 6: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_localize_page.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/app.py collab_splats/dashboard/localize.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): live 'pulling from server' progress on scene/video fetch (F3)"
```

---

## Task 5: Move blocking rclone watchers off the IOLoop (F2)

**Files:**
- Modify: `collab_splats/dashboard/app.py` (`_on_session`; `has_processed` checks in `_autoload_current`/`_on_run`)
- Modify: `collab_splats/dashboard/localize.py` (`_on_scene_session`, `_on_field_session`, `_on_camera`)
- Test: `tests/dashboard/test_app.py`, `tests/dashboard/test_localize_page.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_app.py`:

```python
def test_on_session_lists_videos_off_loop(tmp_path):
    """Selecting a session must not call rclone list_videos on the calling (IOLoop) thread."""
    import threading

    app = _app(tmp_path)  # existing helper
    calling_thread = threading.current_thread().name
    ran_on = {}
    orig = app._source.list_videos

    def tracking_list(sess):
        ran_on["thread"] = threading.current_thread().name
        return orig(sess)

    app._source.list_videos = tracking_list
    app._on_session(type("E", (), {"new": "2026_05_07"})())
    # Give the background thread a moment.
    if getattr(app, "_video_list_thread", None):
        app._video_list_thread.join(timeout=5)
    assert ran_on["thread"] != calling_thread
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py::test_on_session_lists_videos_off_loop -v`
Expected: FAIL — `_on_session` calls `list_videos` inline on the calling thread.

- [ ] **Step 3: Background `_on_session` in app.py**

Replace `_on_session` (lines 280-284):

```python
    def _on_session(self, event) -> None:
        """Populate video dropdown when session changes (rclone list runs off the IOLoop)."""
        if not event.new:
            return
        doc = pn.state.curdoc
        session = event.new

        def work() -> None:
            try:
                videos = self._source.list_videos(session)
            except Exception:
                logger.warning("video listing failed for %s", session, exc_info=True)
                return
            setter = lambda: setattr(self.video_select, "options", videos)
            doc.add_next_tick_callback(setter) if doc is not None else setter()

        self._video_list_thread = threading.Thread(target=work, name="video-list", daemon=True)
        self._video_list_thread.start()
```

- [ ] **Step 4: Background the `has_processed` checks**

`_autoload_current` (line 345) and `_on_run` (line 384) call `has_processed` (blocking rclone) on the IOLoop. `has_processed` becomes memoized in Task 6, but move the network branch off-loop now. In `_autoload_current`, replace the final block:

```python
        out = self._base_dir / session / stem
        if (out / "feedforward.zarr").exists():
            self._load_outputs(session, stem)
            return
        # Remote check is a blocking rclone list -> run off the IOLoop, then load if present.
        doc = pn.state.curdoc

        def work() -> None:
            if self._source.has_processed(session, stem):
                cb = lambda: self._load_outputs(session, stem)
                doc.add_next_tick_callback(cb) if doc is not None else cb()

        threading.Thread(target=work, name="has-processed", daemon=True).start()
```

Leave `_on_run`'s cached-check as is for now (it already runs inside a user click; the memoization in Task 6 removes the repeat cost). Add a code comment at line 384 noting the check is memoized by `SessionSource` (Task 6).

- [ ] **Step 5: Background the localize watchers**

In `collab_splats/dashboard/localize.py`, rewrite `_on_scene_session` (213-216), `_on_field_session` (253-256), and `_on_camera` (258-261) to run their `list_*` call on a daemon thread and set the target widget's `.options` back via `doc.add_next_tick_callback`, mirroring the existing `_on_scene_video` pattern (225-237). Example for `_on_field_session`:

```python
    def _on_field_session(self, event) -> None:
        if not event.new:
            return
        doc = pn.state.curdoc
        fs = event.new

        def work():
            try:
                cams = self._source.list_rgb_cameras(fs)
            except Exception:
                logger.warning("camera listing failed for %s", fs, exc_info=True)
                return
            setter = lambda: setattr(self.camera, "options", cams)
            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="camera-list", daemon=True).start()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_localize_page.py -v`
Expected: PASS. (Existing `test_selecting_session_lists_videos` may need `app._video_list_thread.join(timeout=5)` before asserting `video_select.options` — update it if it now races.)

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/app.py collab_splats/dashboard/localize.py tests/dashboard/test_app.py tests/dashboard/test_localize_page.py
git commit -m "perf(dashboard): move blocking rclone listings off the IOLoop (F2)"
```

---

## Task 6: Memoize rclone listings + has_processed (F7)

**Files:**
- Modify: `collab_splats/dashboard/sources.py` (TTL cache wrapper on listers + `has_processed`; `invalidate` method)
- Test: `tests/dashboard/test_sources.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_sources.py`:

```python
def test_list_sessions_is_memoized():
    client = _client()
    client.list_directory.return_value = [{"Name": "2026_05_07", "IsDir": True}]
    src = SessionSource(client=client)

    src.list_sessions()
    src.list_sessions()
    assert client.list_directory.call_count == 1  # second call served from cache


def test_invalidate_clears_memoized_listing():
    client = _client()
    client.list_directory.return_value = [{"Name": "2026_05_07", "IsDir": True}]
    src = SessionSource(client=client)

    src.list_sessions()
    src.invalidate()
    src.list_sessions()
    assert client.list_directory.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_sources.py::test_list_sessions_is_memoized tests/dashboard/test_sources.py::test_invalidate_clears_memoized_listing -v`
Expected: FAIL — no caching; `list_directory` called twice; `invalidate` undefined.

- [ ] **Step 3: Add a small TTL cache to `SessionSource`**

In `collab_splats/dashboard/sources.py`, add `import time` to imports. In `SessionSource.__init__`, after setting `self._client`, add:

```python
        # (key -> (expiry_epoch, value)) memo for cheap-but-repeated rclone directory listings.
        self._listing_cache: dict = {}
        self._listing_ttl = 60.0  # seconds; bucket contents rarely change mid-session
```

Add a private helper and a public `invalidate` right after `_require_client`:

```python
    def _cached(self, key: tuple, producer):
        """Return a memoized listing for key, refreshing when the TTL has elapsed."""
        now = time.monotonic()
        hit = self._listing_cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        value = producer()
        self._listing_cache[key] = (now + self._listing_ttl, value)
        return value

    def invalidate(self, key: tuple | None = None) -> None:
        """Drop one cached listing (by key) or the whole listing cache."""
        if key is None:
            self._listing_cache.clear()
        else:
            self._listing_cache.pop(key, None)
```

Wrap each lister body in `_cached`. Example for `list_sessions`:

```python
    def list_sessions(self) -> list[str]:
        """Return sorted YYYY_MM_DD session directory names (memoized)."""
        def produce():
            client = self._require_client()
            items = client.list_directory(CURATED_BUCKET, ROOT)
            return sorted(i["Name"] for i in items if i.get("IsDir"))

        return self._cached(("list_sessions",), produce)
```

Apply the same wrapping (with distinct key tuples including args) to `list_videos`, `list_field_sessions`, `list_rgb_cameras`, `list_camera_videos`, `list_localization_dbs`, and `has_processed`.

- [ ] **Step 4: Invalidate after a push in `push_outputs`**

At the end of `push_outputs` (after a successful `copy`), add `self.invalidate(("has_processed", session, stem))` so a freshly-produced scene is visible immediately.

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_sources.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/sources.py tests/dashboard/test_sources.py
git commit -m "perf(dashboard): TTL-memoize rclone listings + has_processed (F7)"
```

---

## Task 7: Cache mesh PolyData; make normalize non-mutating (F11)

**Files:**
- Modify: `collab_splats/dashboard/viewer.py` (`load`, `_render_left`, `_render_right`, `ensure_mesh_features`, `_normalize`)
- Test: `tests/dashboard/test_viewer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_viewer.py` (follow the file's existing fixture style — off-screen `SplitViewer`, a small `FeedforwardResult`-like stub, and a tiny mesh `.ply` written to `tmp_path`):

```python
def test_mesh_read_from_disk_once_across_renders(tmp_path, monkeypatch):
    """Mesh is read once at load and reused; mode/normalize/query don't re-read from disk."""
    import collab_splats.dashboard.viewer as viewer_mod

    reads = {"n": 0}
    real_read = viewer_mod.pv.read

    def counting_read(path):
        reads["n"] += 1
        return real_read(path)

    monkeypatch.setattr(viewer_mod.pv, "read", counting_read)

    v = _viewer_with_scene(tmp_path)  # helper: SplitViewer(off_screen=True) + loaded scene + mesh
    v.set_mode("mesh")
    v.set_mode("pointcloud")
    v.set_mode("mesh")
    v.set_normalize_view(False)
    assert reads["n"] <= 1  # cached after the first load; no re-reads per interaction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py::test_mesh_read_from_disk_once_across_renders -v`
Expected: FAIL — `pv.read` is called on every `_render_left`/`_render_right`.

- [ ] **Step 3: Cache the mesh at load; reuse it**

In `viewer.py`, add `self._mesh_polydata: pv.PolyData | None = None` to `__init__` state. In `load()`, after setting `self._mesh_path`, read once:

```python
        # Read the mesh once here and cache the PolyData; renders reuse it (no per-interaction disk read).
        self._mesh_polydata = pv.read(str(self._mesh_path)) if (self._mesh_path and self._mesh_path.exists()) else None
```

In `_render_left` (line 236) and `_render_right` (line 259), replace `pv.read(str(self._mesh_path))` with a copy of the cached PolyData:

```python
            mesh = self._mesh_polydata.copy()
```

(Use `.copy()` because `_normalize`/`point_data["RGB"]` mutate the object; the cache must stay pristine.)

- [ ] **Step 4: Make `_normalize` non-mutating**

Replace `_normalize` (lines 193-197):

```python
    def _normalize(self, mesh: pv.PolyData) -> pv.PolyData:
        """Return a view-normalized copy of a mesh/cloud (no-op when normalization off).

        Non-mutating: the caller may pass a cached PolyData; transforming in place would
        compound the transform across renders.
        """
        if self._view_T is None:
            return mesh
        return mesh.transform(self._view_T, inplace=False)
```

Since `_render_*` now pass a fresh `.copy()` (mesh) or freshly-built polydata (cloud), `inplace=False` adds one more copy only for the mesh path — acceptable and correct.

- [ ] **Step 5: Reuse the cached mesh in `ensure_mesh_features`**

In `ensure_mesh_features` (line 174), the vertices come from open3d, independent of the render cache — leave the `o3d.io.read_triangle_mesh` as is (different library, needed for `features2vertex`), but add a comment that this is a one-time first-mesh-query cost, not per-render.

- [ ] **Step 6: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -v`
Expected: PASS (existing viewer tests + the new one).

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/viewer.py tests/dashboard/test_viewer.py
git commit -m "perf(dashboard): cache mesh PolyData at load; non-mutating normalize (F11)"
```

---

## Task 8: Recolor the right pane without geometry rebuild (F12)

**Files:**
- Modify: `collab_splats/dashboard/viewer.py` (`render_query` / `_render_right`)
- Test: `tests/dashboard/test_viewer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_viewer.py`:

```python
def test_recolor_updates_scalars_without_rebuilding(tmp_path, monkeypatch):
    """A query recolor updates point RGB in place; it does not clear + rebuild the pane."""
    import numpy as np

    v = _viewer_with_scene(tmp_path)  # pointcloud mode, no mesh
    clears = {"n": 0}
    real_clear = v._right.clear
    monkeypatch.setattr(v._right, "clear", lambda *a, **k: (clears.__setitem__("n", clears["n"] + 1), real_clear(*a, **k))[1])

    n = len(v._result.points)
    colors = np.zeros((n, 3), dtype=np.uint8)
    v.render_query(colors)  # first query: may build
    baseline = clears["n"]
    colors2 = np.full((n, 3), 7, dtype=np.uint8)
    v.render_query(colors2)  # second recolor: must NOT clear again
    assert clears["n"] == baseline
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py::test_recolor_updates_scalars_without_rebuilding -v`
Expected: FAIL — `_render_right` calls `self._right.clear()` every time.

- [ ] **Step 3: Add an in-place recolor fast path**

In `viewer.py`, keep a handle to the right cloud's polydata. Add `self._right_cloud: pv.PolyData | None = None` to `__init__`. In `_render_right`'s pointcloud branch, store the built cloud: `self._right_cloud = cloud`. Then add a fast path used by `render_query`:

```python
    def render_query(self, colors: np.ndarray) -> None:
        """Recolour the right pane with precomputed query colours (IOLoop thread).

        Fast path (pointcloud mode, same geometry already shown): update the existing
        PolyData's RGB scalars in place instead of clearing + rebuilding the whole scene.
        """
        if (
            self.mode != "mesh"
            and self._right_cloud is not None
            and self._display_idx is not None
            and len(self._display_idx) == self._right_cloud.n_points
        ):
            idx = self._display_idx
            self._right_cloud["RGB"] = np.ascontiguousarray(colors[idx]).astype(np.uint8)
            self._right_cloud.Modified()
            if not self._off_screen:
                self._right_pane.synchronize()
            return
        self._render_right(colors)
```

Ensure `pointcloud_to_polydata` attaches an `"RGB"` array the mapper renders (it already sets RGB per `viz_utils`); if the active scalar name differs, set `self._right_cloud.set_active_scalars("RGB")` once when built.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/viewer.py tests/dashboard/test_viewer.py
git commit -m "perf(dashboard): in-place point recolor on query, no geometry rebuild (F12)"
```

---

## Task 9: SceneCache for SplatsApp (F15)

**Files:**
- Modify: `collab_splats/dashboard/shell.py` (pass shared `cache` to `SplatsApp`)
- Modify: `collab_splats/dashboard/app.py` (accept `cache`; short-circuit reselect; cache result/mesh/lifted; invalidate on force/run)
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_app.py`:

```python
def test_reselecting_loaded_scene_skips_reload(tmp_path):
    """Reselecting the already-displayed scene must not enqueue another load job."""
    from collab_splats.dashboard.localize import SceneCache

    app = _recording_app(tmp_path, cache=SceneCache())
    app._current_scene = ("2026_05_07", "clip_03")  # pretend it is displayed
    before = len(app._gpu.submitted)
    app._load_outputs("2026_05_07", "clip_03")
    assert len(app._gpu.submitted) == before  # short-circuited, no new job


def test_force_run_invalidates_scene_cache(tmp_path):
    from collab_splats.dashboard.localize import SceneCache

    cache = SceneCache()
    cache.put(("2026_05_07", "clip_03"), "result", object())
    app = _recording_app(tmp_path, cache=cache)
    app._invalidate_scene("2026_05_07", "clip_03")
    assert cache.get(("2026_05_07", "clip_03"), "result") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py::test_reselecting_loaded_scene_skips_reload tests/dashboard/test_app.py::test_force_run_invalidates_scene_cache -v`
Expected: FAIL — `SplatsApp` has no `cache`, `_current_scene`, or `_invalidate_scene`.

- [ ] **Step 3: Accept the cache in `SplatsApp.__init__`**

Add `cache=None` to the `__init__` signature and store it:

```python
        from collab_splats.dashboard.localize import SceneCache

        self._cache = cache if cache is not None else SceneCache()
        self._current_scene: tuple | None = None
```

(Import `SceneCache` at the top of `app.py` instead of inline if no circular-import issue; `localize.py` imports from `app.py` via `shell.py` only, so a top-level import here is safe — verify with a quick `python -c "import collab_splats.dashboard.app"`.)

- [ ] **Step 4: Pass the shared cache from the shell**

In `collab_splats/dashboard/shell.py:34`, pass the cache:

```python
        self._splats = SplatsApp(
            base_dir=Path(base_dir), source=source, gpu_worker=gpu_worker, op_log=op_log, cache=self._cache
        )
```

- [ ] **Step 5: Short-circuit + cache in `_load_outputs`; add invalidation**

At the top of `_load_outputs`, short-circuit an already-displayed scene:

```python
        if self._current_scene == (session, stem) and not self._op_log.is_running:
            return
```

In the load `job()`, consult/populate the cache: return `self._cache.get((session, stem), "result")` when present instead of re-reading; otherwise after `load_zarr`, `self._cache.put((session, stem), "result", result)` (and similarly cache `mesh_path`/`lifted_normed`). In `on_done`, set `self._current_scene = (session, stem)`.

Add the invalidation helper:

```python
    def _invalidate_scene(self, session: str, stem: str) -> None:
        """Drop cached loads for a scene (used after Force re-run / fresh pipeline output)."""
        for kind in ("result", "mesh", "lifted_normed"):
            self._cache.put((session, stem), kind, None)
        if self._current_scene == (session, stem):
            self._current_scene = None
        self._source.invalidate(("has_processed", session, stem))
```

Call `self._invalidate_scene(session, stem)` in `_on_run` when `force=True` and in `on_done` after a successful `run_pipeline`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_shell.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/app.py collab_splats/dashboard/shell.py tests/dashboard/test_app.py
git commit -m "perf(dashboard): share SceneCache with SplatsApp; skip reselect reload (F15)"
```

---

## Task 10: Warm the full heavy stack (F1)

**Files:**
- Modify: `collab_splats/dashboard/app.py:604-617` (`_warm_heavy_stack`)
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_app.py`:

```python
def test_warm_heavy_stack_imports_localizer_and_pipeline(monkeypatch):
    """Warm thread must front-load the localizer + mesh/pipeline stacks, not just feedforward."""
    import importlib

    from collab_splats.dashboard import app as app_mod

    imported = []
    real_import = importlib.import_module

    def tracking_import(name, *a, **k):
        imported.append(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(app_mod.importlib, "import_module", tracking_import, raising=False)
    app_mod._warm_heavy_stack()
    assert any("localization.localizer" in n for n in imported)
    assert any("dashboard.pipeline" in n for n in imported)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py::test_warm_heavy_stack_imports_localizer_and_pipeline -v`
Expected: FAIL — current warm imports only `feedforward.base` + `semantics.features.base` via direct `import`, and uses no `importlib`.

- [ ] **Step 3: Warm via importlib so the set is testable + complete**

Add `import importlib` to `app.py` imports. Replace `_warm_heavy_stack` (604-617):

```python
def _warm_heavy_stack() -> None:
    """Import the heavy reconstruction/semantics/localization stack once at startup (bg thread).

    Pre-pays the ~17s import + torch.compile so the first Run/load/query/Localize isn't a cold
    start. Runs off the IOLoop (the server already binds and the page renders before this finishes).
    """
    modules = (
        "collab_splats.dashboard.pipeline",  # pulls feedforward + mesh/TSDF
        "collab_splats.semantics.features.base",
        "collab_splats.localization.localizer",  # localize tab's first run
    )
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            logger.warning("heavy-stack warm failed for %s: %s", name, exc)
    logger.info("heavy stack warmed")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "perf(dashboard): warm localizer + mesh/pipeline stack on launch (F1)"
```

---

# PHASE 2 — Display tuning & decode deferral (low risk)

## Task 11: Cheaper point rendering (F13)

**Files:**
- Modify: `collab_splats/dashboard/viz_utils.py` (`PCD_KWARGS`)
- Test: `tests/dashboard/test_viz_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_viz_utils.py`:

```python
def test_pcd_kwargs_uses_flat_points():
    """Spheres at point_size 0.5 are invisible; render as flat GL points to save VTK cost."""
    from collab_splats.dashboard.viz_utils import PCD_KWARGS

    assert PCD_KWARGS.get("render_points_as_spheres") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viz_utils.py::test_pcd_kwargs_uses_flat_points -v`
Expected: FAIL — currently `render_points_as_spheres=True`.

- [ ] **Step 3: Flip to flat points**

In `collab_splats/dashboard/viz_utils.py`, in the `PCD_KWARGS` dict set `render_points_as_spheres=False`. Keep `point_size` (bump to `2.0` so flat points remain visible). Add a comment: `# flat GL points: far cheaper than spheres for 500k points, and spheres at this size were invisible anyway`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viz_utils.py tests/dashboard/test_viewer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/viz_utils.py tests/dashboard/test_viz_utils.py
git commit -m "perf(dashboard): render pointcloud as flat GL points not spheres (F13)"
```

---

## Task 12: Skip redundant per-render camera/light work (F14)

**Files:**
- Modify: `collab_splats/dashboard/viewer.py` (`_apply_view` guarded; recolor path already skips it via Task 8)
- Test: `tests/dashboard/test_viewer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_viewer.py`:

```python
def test_recolor_fast_path_does_not_reapply_view(tmp_path, monkeypatch):
    """The in-place recolor path must not re-run camera/light setup."""
    import numpy as np

    v = _viewer_with_scene(tmp_path)  # pointcloud mode
    calls = {"n": 0}
    monkeypatch.setattr(v, "_apply_view", lambda plotter: calls.__setitem__("n", calls["n"] + 1))

    n = len(v._result.points)
    v.render_query(np.zeros((n, 3), dtype=np.uint8))  # builds once
    v.render_query(np.ones((n, 3), dtype=np.uint8))   # fast path
    # Fast path recolor adds no _apply_view calls beyond the initial build.
    assert calls["n"] <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py::test_recolor_fast_path_does_not_reapply_view -v`
Expected: PASS already if Task 8's fast path returns before `_apply_view` — if it FAILS, the fast path is calling `_render_right`. Fix so the fast path in `render_query` (Task 8, Step 3) does not call `_apply_view` (it does not, by construction). This task's test is the regression guard; if it passes immediately, note that and proceed to Step 5.

- [ ] **Step 3: (If needed) ensure the fast path skips `_apply_view`**

Confirm the `render_query` fast path from Task 8 returns after `synchronize()` without calling `_render_right`/`_apply_view`. No code change if already correct.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/viewer.py tests/dashboard/test_viewer.py
git commit -m "perf(dashboard): recolor fast path skips camera/light reapply (F14)"
```

---

## Task 13: Defer `lifted_normed.npy` load to first query (F10)

**Files:**
- Modify: `collab_splats/dashboard/app.py:429-463` (`_load_outputs` stops eager `np.load`; viewer lifts lazily)
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_app.py`:

```python
def test_load_does_not_eager_load_lifted_normed(tmp_path, monkeypatch):
    """The display load must not np.load lifted_normed.npy before any query is issued."""
    import numpy as np

    import collab_splats.dashboard.app as app_mod

    loaded = {"n": 0}
    monkeypatch.setattr(app_mod.np, "load", lambda *a, **k: loaded.__setitem__("n", loaded["n"] + 1))

    app = _recording_app(tmp_path)
    # Drive the load job's numpy usage indirectly: assert no np.load in the job body.
    app._load_outputs("2026_05_07", "clip_03")
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    try:
        job_fn()
    except Exception:
        pass
    assert loaded["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py::test_load_does_not_eager_load_lifted_normed -v`
Expected: FAIL — `_load_outputs` calls `np.load(lifted_path)` at line 441.

- [ ] **Step 3: Drop the eager load; rely on lazy lift**

In `_load_outputs`'s `job()`, remove the `lifted_path`/`np.load` block (440-441) and pass `lifted_normed=None`. The viewer already lifts lazily on first query via `ensure_lifted` (`viewer.py:144`), which reads the cached features when needed. Update the returned tuple + `self._viewer.load(...)` call to pass `lifted_normed=None`. Keep `semantics_dir` so `ensure_lifted` can find the features.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_viewer_lift.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "perf(dashboard): defer lifted-feature load to first query (F10)"
```

---

## Task 14: Opt-in dense-array decode in `load_zarr` (F8)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:177-235` (`load_zarr` gains per-array flags)
- Modify: `collab_splats/dashboard/app.py` (display load passes the lean flag set)
- Test: `tests/pointcloud/feedforward/test_load_zarr_flags.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/pointcloud/feedforward/test_load_zarr_flags.py`:

```python
"""load_zarr must be able to skip decoding dense optional arrays for the display path."""
import numpy as np

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _tiny_result_with_dense(tmp_path):
    """Minimal FeedforwardResult with points/colors/extrinsics + a dense depth array; save_zarr it."""
    n, p, h, w = 2, 5, 4, 4
    result = FeedforwardResult(
        points=np.zeros((p, 3), dtype=np.float32),
        colors=np.zeros((p, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        image_paths=[tmp_path / f"{i:05d}.jpg" for i in range(n)],
        original_coords=np.zeros((n, 6), dtype=np.float32),
        model_width=w,
        model_height=h,
        depth=np.ones((n, h, w), dtype=np.float32),  # the dense array under test
    )
    store = tmp_path / "feedforward.zarr"
    result.save_zarr(store)
    return store


def test_load_zarr_can_skip_depth(tmp_path):
    store = _tiny_result_with_dense(tmp_path)
    lean = FeedforwardResult.load_zarr(store, load_depth=False)
    assert lean.depth is None  # skipped, not decoded
    full = FeedforwardResult.load_zarr(store)
    assert full.depth is not None  # default unchanged (back-compat)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_load_zarr_flags.py -v`
Expected: FAIL — `load_zarr` has no `load_depth` parameter.

- [ ] **Step 3: Add per-array flags to `load_zarr`**

In `collab_splats/pointcloud/feedforward/base.py`, extend the `load_zarr` signature (mirroring the existing `load_images=False` gate) with `load_depth=True, load_world_points=True, load_confidence=True, load_features=True, load_pixel_indices=True`, all defaulting to current behavior. Guard each optional-array read (lines 206-211) behind its flag, e.g.:

```python
        depth = store["depth"][:] if (load_depth and "depth" in store) else None
```

Keep required arrays (points/colors/extrinsics/intrinsics) always loaded.

- [ ] **Step 4: Display path passes the lean set**

In `collab_splats/dashboard/app.py` `_load_outputs` `job()`, load lean:

```python
            result = FeedforwardResult.load_zarr(
                out / "feedforward.zarr",
                load_depth=False,
                load_world_points=False,
                load_confidence=False,
                load_features=False,
                load_pixel_indices=False,
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_load_zarr_flags.py tests/dashboard/test_app.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py collab_splats/dashboard/app.py tests/pointcloud/feedforward/test_load_zarr_flags.py
git commit -m "perf(dashboard): opt-in dense-array decode in load_zarr; lean display load (F8)"
```

---

## Task 15: Debounce state persistence (F16)

**Files:**
- Modify: `collab_splats/dashboard/app.py:107-114,190-191` (`_persist_state` debounced)
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_app.py`:

```python
def test_persist_state_is_debounced(tmp_path, monkeypatch):
    """Rapid widget changes coalesce into a single disk write, not one per event."""
    app = _app(tmp_path)
    writes = {"n": 0}
    monkeypatch.setattr(type(app._state_path), "write_text", lambda self, text: writes.__setitem__("n", writes["n"] + 1))

    # Simulate 5 rapid changes; with debounce only a timer-flush should write.
    for _ in range(5):
        app._persist_state()
    assert writes["n"] == 0  # nothing written synchronously; a scheduled flush does it
    app._flush_state()       # explicit flush (what the debounce timer calls)
    assert writes["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py::test_persist_state_is_debounced -v`
Expected: FAIL — `_persist_state` writes synchronously every call; `_flush_state` undefined.

- [ ] **Step 3: Split persist into mark-dirty + flush**

In `app.py`, add a dirty flag in `__init__`: `self._state_dirty = False`. Replace `_persist_state` so it only marks dirty and schedules a flush on the IOLoop (a Panel periodic/`add_timeout` — or `pn.state.add_periodic_callback` guarded like `main()`); extract the actual write into `_flush_state`:

```python
    def _persist_state(self, *_event) -> None:
        """Mark UI state dirty; the debounce flush writes it (coalesces rapid changes)."""
        self._state_dirty = True

    def _flush_state(self) -> None:
        """Write current widget values to disk if dirty (called by the debounce timer / on run)."""
        if not self._state_dirty:
            return
        self._state_dirty = False
        data = {k: w.value for k, w in self._persisted.items()}
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(yaml.safe_dump(data, sort_keys=False))
        except Exception:
            logger.warning("could not persist dashboard state", exc_info=True)
```

In `main()` (where the periodic poll is set up), also register a slower `_flush_state` periodic (e.g. `period=1000`) guarded by the same `try/except` as the progress tick. Call `self._flush_state()` directly at the start of `_on_run` so state is durable before a run.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "perf(dashboard): debounce dashboard-state disk writes (F16)"
```

---

## Task 16: Lazy plotter build for the inactive tab (F4)

**Files:**
- Modify: `collab_splats/dashboard/shell.py` (build inactive page's `main()` on first activation)
- Test: `tests/dashboard/test_shell.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_shell.py`:

```python
def test_inactive_tab_main_built_lazily(tmp_path, monkeypatch):
    """The Localize page's main() (VTK plotter) is not built until its tab is first shown."""
    import collab_splats.dashboard.shell as shell_mod

    built = {"localize_main": 0}
    real_main = shell_mod.LocalizePage.main

    def counting_main(self):
        built["localize_main"] += 1
        return real_main(self)

    monkeypatch.setattr(shell_mod.LocalizePage, "main", counting_main)

    shell = shell_mod.DashboardShell(base_dir=tmp_path)
    shell.view()  # Splats active by default
    assert built["localize_main"] == 0  # localize main deferred
    shell._on_tab(type("E", (), {"new": 1, "old": 0})())  # switch to Localize
    assert built["localize_main"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_shell.py::test_inactive_tab_main_built_lazily -v`
Expected: FAIL — `view()` calls both `self._splats.main()` and `self._localize.main()` eagerly in the `pn.Tabs(...)` constructor.

- [ ] **Step 3: Defer the inactive tab's `main()`**

In `shell.py:view()`, build the Localize tab with a placeholder and populate it on first `_on_tab` switch. Replace the tabs construction:

```python
        self._localize_built = False
        self._localize_holder = pn.Column(sizing_mode="stretch_both")
        self._tabs = pn.Tabs(
            ("Splats", self._splats.main()),
            ("Localize", self._localize_holder),
            dynamic=True,
            sizing_mode="stretch_both",
        )
```

Extend `_on_tab` to build once on first activation:

```python
    def _on_tab(self, event) -> None:
        """Swap sidebar to the active tab; build the localize view lazily; free GPU on leave."""
        if event.new == 1 and not self._localize_built:
            self._localize_holder[:] = [self._localize.main()]
            self._localize_built = True
        page = self._splats if event.new == 0 else self._localize
        self._sidebar_holder[:] = [page.sidebar()]
        if event.old == 1:
            self._localize.release_gpu()
```

Update the shell.py:48-51 docstring to state the inactive tab's main/plotter is now built on first activation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_shell.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/shell.py tests/dashboard/test_shell.py
git commit -m "perf(dashboard): build inactive tab's VTK view lazily on first activation (F4)"
```

---

# Final verification

- [ ] **Full dashboard suite:**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ tests/pointcloud/feedforward/ -v`
Expected: all PASS.

- [ ] **Format:**

Run: `black collab_splats/dashboard collab_splats/pointcloud/feedforward tests/dashboard tests/pointcloud/feedforward && isort collab_splats/dashboard collab_splats/pointcloud/feedforward tests/dashboard tests/pointcloud/feedforward`

- [ ] **Manual smoke (tmux, not notebook):** launch `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard`, then in a browser confirm:
  - server binds fast; page renders before the warm log line appears
  - selecting a remote scene shows a live "⬇ pulling from server N%" bar (was frozen)
  - Localize tab opens without a ~10s cold stall
  - toggling pointcloud↔mesh / normalize / running a query is snappy (no mesh re-read); reselecting the same scene is instant
  - closing the tab mid-Run then reopening still yields a working dashboard (worker alive)

- [ ] **Update graph:** `graphify update .`

- [ ] **Update CLAUDE.md In-Flight Work** — mark `dashboard-loadtime` done or move to a completed section, referencing this plan + the spec.
