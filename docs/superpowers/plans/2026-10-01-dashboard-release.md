# Dashboard Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the dashboard loads and runs scenes through `Reconstructor`, so two backends (e.g. `vggt_omega`, `instantsfm`) over one `<scene>/images/` can be switched in the local `SplitViewer`.

**Architecture:** almost all deletion. The dashboard's own pipeline copy (`pipeline.run_pipeline`, `RunConfig`, lift/AE helpers), the localization tab and the tabbed shell go. `SplatsApp` builds a `Reconstructor` per `(scene, backend)` from the recorded run config, then calls `done` / `outputs` / `result` / `run`. The viewer reads the lifted store with `semantics.store.read_point_features`.

**Tech Stack:** Panel, PyVista, `collab_splats.reconstructor.Reconstructor`, `semantics.store`, pytest.

**Spec:** [2026-10-01-dashboard-release-design.md](../specs/2026-10-01-dashboard-release-design.md)

---

## Conventions for every task

- **Worktree:** `/workspace/collab-splats/.worktrees/dashboard-release` (`WT` below). Every command runs there.
- **Test command:** `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest <target> -q`. Never `| tail`, because it hides the exit code.
- **Commit:** list every path and use `git commit --only <paths>`; docs need `git add -f`. End the message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Smoke gate:** `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke` must exit 0 before every commit that touches `collab_splats/dashboard/`.
- **Containment:** edits go only in `collab_splats/dashboard/`, `tests/dashboard/`, and `docs/superpowers/`.
- **Comments:** one plain line per block comment, saying what the code does (no header+bullet runs). Use `rtk proxy grep` when grep output looks mangled.

## Deviations from the spec (decided while planning)

| Spec says | Plan does | Why |
|---|---|---|
| `semantics.store.load_point_features` | `read_point_features` | the real name in `semantics/store.py` |
| viewer deletes `load_mesh_vertex_features` | deletes `read_mesh_vertex_features` | the real name; `ensure_mesh_features` transfers on demand |
| `PULL_EXCLUDES` gets a `*/` prefix | moved verbatim | rclone excludes without a leading `/` already match at any depth |
| helper `_backend_config` | folded into `_reconstructor(scene, backend, overrides)` | load and run both need the `Reconstructor`, not just the dict |
| force checkbox | keeps the existing Force re-run button → `overwrite=True` | existing widget, no new code |
| `_scan_output_dirs` gates on `done("preproc")` | `_scan_output_dirs` deleted | it has no caller; the scene list comes from `SceneSource` |
| `test_pipeline.py` moves onto the fixture | deleted | every test in it targets deleted code; `_push_async` keeps no test of its own (it is unchanged) |
| empty backend dropdown when nothing is done | dropdown always lists every registered backend; status line says "not run" | no discovery pass (user: "is all of this new code necessary") |
| — | Run's remote cache-check token logic deleted | Run now always runs; the load path owns the pull |
| — | the console moves from `shell.py` into `SplatsApp.view()` | the shell is gone, and the console is the run-progress surface |

## File map

| File | After this plan |
|---|---|
| `collab_splats/dashboard/app.py` | `SplatsApp` over `Reconstructor`; gains `SceneCache`, `PULL_EXCLUDES`, `_reconstructor`; loses `RunConfig` widgets, lift helpers, `_warm_heavy_stack`, `run_app`, `SplatsPage`, `_scan_output_dirs` |
| `collab_splats/dashboard/viewer.py` | `load(..., lifted_store=)`; `ensure_lifted` reads the store; lift/save/vertex-npy helpers deleted |
| `collab_splats/dashboard/pipeline.py` | `_push_async` only |
| `collab_splats/dashboard/serve.py` | factory serves `SplatsApp(...).view()`; warm list updated |
| `collab_splats/dashboard/{localize,shell,config}.py` | deleted |
| `tests/dashboard/test_app.py`, `test_viewer.py` | adapted |
| `tests/dashboard/test_{localize_page,run_localization,localization_config,shell,config,viewer_lift,semantics_layout,pipeline}.py` | deleted |

---

### Task 0: Rebase onto `rebase/ocr-lens` (GATED: needs the user's go-ahead)

The user said "don't do anything w/ ocr-lens yet". Do not start this task until the user explicitly says to. Tasks 1–5 need `semantics/store.py` and `transfer_features`, and both exist only on `rebase/ocr-lens`.

**Files:** none (branch operation)

- [ ] **Step 1: Confirm both tips**

```bash
cd $WT && git log --oneline -1 clean/dashboard-release && git log --oneline -1 rebase/ocr-lens
git status --short   # must be empty except untracked docs
```

- [ ] **Step 2: Back up, then rebase the spec/plan commits**

```bash
git update-ref refs/backup/dashboard-release/pre-ocr-rebase clean/dashboard-release
git rebase rebase/ocr-lens
```

Expected: no conflicts, because the branch holds only docs commits.

- [ ] **Step 3: Baseline gate**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard -q`
Expected: all pass. Record the count in the task report. Every later count is compared against it.

---

### Task 1: Drop the localization tab and the tabbed shell

**Files:**
- Delete: `collab_splats/dashboard/localize.py`, `collab_splats/dashboard/shell.py`
- Delete: `tests/dashboard/test_localize_page.py`, `test_run_localization.py`, `test_localization_config.py`, `test_shell.py`
- Modify: `collab_splats/dashboard/app.py` (imports, `SceneCache`, `view`, entry points)
- Modify: `collab_splats/dashboard/pipeline.py` (delete the localization block, line ~404 to EOF)
- Modify: `collab_splats/dashboard/config.py` (delete `LocalizationConfig`)
- Modify: `collab_splats/dashboard/serve.py` (`_WARM_MODULES`, `make_factory`)
- Modify: `tests/dashboard/test_app.py`, `tests/dashboard/test_pipeline.py`

- [ ] **Step 1: Write the failing test for the console in `view()`**

Add to `tests/dashboard/test_app.py`:

```python
def test_view_hosts_the_operations_console(tmp_path):
    """The op-log console renders under the viewer now that the tabbed shell is gone."""
    app, _ = _app(tmp_path)
    app._op_log.append_line("hello console")
    app.view()
    app._on_progress_tick()
    assert "hello console" in app._progress.object
```

Change the import at the top of `test_app.py`:

```python
from collab_splats.dashboard.app import SceneCache, SplatsApp
```

(and delete `from collab_splats.dashboard.localize import SceneCache`).

Delete these tests from `test_app.py`: `test_run_app_serves_with_hardening`, `test_warm_heavy_stack_imports_localizer_and_pipeline`.

In `test_force_run_invalidates_scene_cache` and `test_view_mode_mesh_populates_shared_cache`, delete the comment words that mention LocalizePage and keep the assertions.

- [ ] **Step 2: Run it and confirm it fails**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q`
Expected: collection error, `ImportError: cannot import name 'SceneCache' from 'collab_splats.dashboard.app'`.

- [ ] **Step 3: Move `SceneCache` into `app.py`**

Paste the class from `localize.py` (line ~86) into `app.py`, just above `class SplatsApp`. Drop `drop_kind` and `clear` (no callers once localize is gone; confirm with `rtk proxy grep -rn "drop_kind\|\.clear()" collab_splats/dashboard tests/dashboard`). The docstring and `_KIND_KEEP` comment then become:

```python
class SceneCache:
    """Session-level cache of expensive loads, keyed (scene_key, kind): 'loaded' tuples and 'mesh' polydata."""

    # Mesh polydata keeps the last 3; "loaded" is bounded by SplatsApp._remember_loaded
    _KIND_KEEP = {"mesh": 3}
```

Leave `__init__`, `get`, `put`, `drop`, `drop_scene` verbatim. Add `from collections import deque` if it isn't already imported (it is). Replace the `from collab_splats.dashboard.localize import SceneCache` import line with nothing.

In `SplatsApp.__init__`, change the cache comment to `# Session-level cache of expensive loads`.

In `_invalidate_scene`, replace the two LocalizePage comment lines with `# Drop every cached kind for the scene`.

In `_on_view_mode`'s job, replace the two "share it with LocalizePage" comment lines with `# Mesh mode: materialise the polydata off the IOLoop; the SceneCache skips a re-read`.

- [ ] **Step 4: Move the console into `view()` and delete the old entry points**

Replace `main()` and `view()` in `app.py` with:

```python
    def main(self) -> pn.Column:
        """Main-area contents: the split viewer."""
        # Mirror the worker's busy flag onto the widgets
        try:
            pn.state.add_periodic_callback(self._sync_busy, period=300, start=True)
        except Exception:
            logger.debug("no periodic callback (no server doc)", exc_info=True)

        # Debounced state flush, plus a final flush on tab close
        try:
            pn.state.add_periodic_callback(self._flush_state, period=1000, start=True)
            pn.state.on_session_destroyed(lambda _ctx: self._flush_state())
        except Exception:
            logger.debug("no periodic callback (no server doc); state flushes on run only", exc_info=True)

        return pn.Column(self._viewer.layout, sizing_mode="stretch_both")

    def view(self) -> pn.template.MaterialTemplate:
        """Page: sidebar, split viewer, and the operations console pinned under it."""
        # Resizable console showing the shared op log
        self._progress = pn.pane.HTML(self._op_log.render_html(), sizing_mode="stretch_both")
        self._console = pn.Column(
            self._progress,
            sizing_mode="stretch_width",
            height=190,
            styles={
                "resize": "vertical",
                "overflow": "auto",
                "min-height": "70px",
                "border-top": "2px solid #2596be",
                "background": "#0d1117",
                "padding": "4px 8px",
            },
        )
        self._seen_log_version = -1

        try:
            pn.state.add_periodic_callback(self._on_progress_tick, period=300, start=True)
        except Exception:
            logger.debug("no periodic callback (no server doc); progress is static", exc_info=True)

        return pn.template.MaterialTemplate(
            title="splats",
            sidebar=[self.sidebar()],
            main=[pn.Column(self.main(), self._console, sizing_mode="stretch_both")],
            header_background="#2596be",
            sidebar_width=340,
        )

    def _on_progress_tick(self) -> None:
        """Refresh the console when the op log changed (300 ms poll, version-gated)."""
        if self._op_log.version != self._seen_log_version:
            self._seen_log_version = self._op_log.version
            self._progress.object = self._op_log.render_html()
```

Change the `sidebar()` docstring to `"""Sidebar contents."""`.

Delete from `app.py`:
- the `SplatsPage = SplatsApp` alias and its comment
- `_warm_heavy_stack`
- `run_app` (`__init__.py` maps `run_app` to `serve.run_app`; `__main__` uses serve)
- the `importlib` import, which has no other user (check with `rtk proxy grep -n importlib collab_splats/dashboard/app.py`)

Keep `_ensure_display` (`serve._finalize` calls it). Its `"########\n# Entry points\n########"` divider stays.

- [ ] **Step 5: Delete the localization files, block and config**

```bash
cd $WT && git rm -q collab_splats/dashboard/localize.py collab_splats/dashboard/shell.py \
  tests/dashboard/test_localize_page.py tests/dashboard/test_run_localization.py \
  tests/dashboard/test_localization_config.py tests/dashboard/test_shell.py
```

In `pipeline.py`:
- Delete everything from `class LocalizationRunOutput` (line ~404) to end of file: `_load_feedforward_result`, `_stamp_db_provenance`, `_build_localizer`, `_resolve_query_intrinsics`, `BrowseData`, `read_localized_group`, `_local_ref_path`, `load_browse_data`, `run_localization`.
- Then delete the imports nothing else uses: `LocalizationConfig` from the config import, and any of `invert_poses`, `PIL.Image`, `read_image`, `to_uint8_hwc`, `functools`, `dataclass` that are now unused. Verify with `/opt/venv/reconstruction/bin/python -m pyflakes collab_splats/dashboard/pipeline.py`; expect no "imported but unused".

In `config.py`, delete `class LocalizationConfig` and any import only it used.

In `test_pipeline.py`, delete `_make_localized_zarr` and every test below it (`test_read_localized_group_*`, `test_load_browse_data_*`, `test_stamp_db_provenance_*`).

- [ ] **Step 6: Point serve at `SplatsApp`**

In `serve.py`, make `_WARM_MODULES`:

```python
_WARM_MODULES = (
    ("torch", "torch runtime"),
    ("collab_splats.dashboard.app", "dashboard core (viewer + models)"),
    ("collab_splats.dashboard.pipeline", "reconstruction pipeline"),
    ("collab_splats.semantics.features.base", "semantic extractors"),
)
```

Then make the `make_factory` inner `factory` this:

```python
    def factory() -> pn.template.MaterialTemplate:
        if not state.ready:
            return _loading_page(state, op_log)

        # Import stays local: app pulls the heavy stack, guaranteed warm here
        from collab_splats.dashboard.app import SplatsApp

        return SplatsApp(base_dir=Path(base_dir), gpu_worker=state.gpu_worker, op_log=op_log).view()
```

Change the `make_factory` docstring "real shell" to "real page".

- [ ] **Step 7: Run the dashboard tests**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard -q`
Expected: all pass, including `test_view_hosts_the_operations_console` and `test_serve.py::test_factory_serves_loading_page_until_ready`.

Run: `rtk proxy grep -rn "localize\|LocalizePage\|DashboardShell\|shell" collab_splats/dashboard tests/dashboard`
Expected: no hits, apart from the `async_utils.py` comment "the localize page"; reword it to "every session".

- [ ] **Step 8: Smoke, then commit**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
git commit --only collab_splats/dashboard tests/dashboard \
  -m "refactor(dashboard): drop the localization tab and tabbed shell

SceneCache moves into app.py; the op-log console moves into SplatsApp.view().

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Viewer reads the lifted store

**Files:**
- Modify: `collab_splats/dashboard/viewer.py`
- Modify: `collab_splats/dashboard/app.py` (`_load_outputs.on_done` kwarg rename only)
- Delete: `tests/dashboard/test_viewer_lift.py`
- Modify: `tests/dashboard/test_viewer.py`, `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing tests**

In `tests/dashboard/test_viewer.py`:
- Delete `test_score_query_lazily_lifts_on_first_query`, `test_score_query_blank_positive_does_not_lift`, and every `test_read_mesh_vertex_features_*`.
- Replace `test_ensure_lifted_uses_cached_lifted_store_fast_path` with the two tests below. They use the file's existing `SplitViewer(off_screen=True)` + `_FakeResult` pattern.

```python
def test_ensure_lifted_reads_the_lifted_store(tmp_path):
    """First query decodes <extractor>_lifted.zarr through the autoencoder stored inside it."""
    store = tmp_path / "talk2dino_lifted.zarr"
    codes = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
    write_point_features(store, codes, FeatureAutoencoder(32, 8))
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(20), mesh_path=None, lifted_store=store)
    v.ensure_lifted()
    assert v._point_features.shape == (20, 32)
    assert np.allclose(np.linalg.norm(v._point_features, axis=1), 1.0, atol=1e-5)


def test_score_query_blank_positive_does_not_read_store(tmp_path, monkeypatch):
    """An empty positive query shows plain RGB without touching the lifted store."""
    reads = []
    monkeypatch.setattr("collab_splats.dashboard.viewer.read_point_features", reads.append)
    v = SplitViewer(off_screen=True)
    result = _FakeResult(20)
    v.load(result, mesh_path=None, lifted_store=tmp_path / "talk2dino_lifted.zarr")
    colors = v.score_query(positive=[], negative=[], extractor_name="talk2dino")
    assert np.array_equal(colors, result.colors)
    assert reads == []
```

Import to add at the top of `test_viewer.py` (`FeatureAutoencoder` is already imported there):

```python
from collab_splats.semantics.store import write_point_features
```

Then drop `import zarr` if no remaining test uses it (`pyflakes tests/dashboard/test_viewer.py`).

Delete `tests/dashboard/test_viewer_lift.py`: `git rm -q tests/dashboard/test_viewer_lift.py`.

In `tests/dashboard/test_app.py::test_load_outputs_on_done_renders_into_viewer`, change the assertion `kwargs["semantics_dir"] == "semdir"` to `kwargs["lifted_store"] == "semdir"`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py tests/dashboard/test_app.py -q`
Expected: FAIL with `TypeError: load() got an unexpected keyword argument 'lifted_store'` and `AttributeError: ... has no attribute 'read_point_features'`.

- [ ] **Step 3: Implement in `viewer.py`**

1. Delete `lift_point_features`, `_save_point_features`, `read_mesh_vertex_features` and the "Heavy semantics imports deferred" comment run. Keep `_decimate_indices`.
2. Add the top-level imports (our-code group, after `viz_utils`):

```python
from collab_splats.semantics.lifting import transfer_features
from collab_splats.semantics.store import read_point_features
```

3. In `__init__`, rename `self._semantics_dir: Path | None = None` to `self._lifted_store: Path | None = None`.
4. Replace `load`'s signature, docstring and the stash line:

```python
    def load(
        self,
        result,
        mesh_path: Path | None,
        point_features: np.ndarray | None = None,
        lifted_store: Path | None = None,
        max_points: int = 500_000,
    ) -> None:
        """Load a PointcloudResult (+ optional mesh) into both panes; features are read on first query."""
```

Then, in the body, replace `self._semantics_dir = Path(semantics_dir) if semantics_dir else None` with:

```python
        self._lifted_store = Path(lifted_store) if lifted_store else None
```

5. In `ensure_mesh_polydata`, delete the "Per-vertex mesh features" comment and the `self._mesh_vertex_features = read_mesh_vertex_features(...)` line. `ensure_mesh_features` fills it on the first mesh query.
6. Replace the whole `ensure_lifted` body:

```python
    def ensure_lifted(self, op_log=None) -> None:
        """Read the lifted per-point features on first query (worker thread)."""
        if self._point_features is not None or self._lifted_store is None:
            return

        if op_log is not None:
            op_log.append_line("query: loading lifted point features")

        self._point_features = read_point_features(self._lifted_store)
```

7. In `ensure_mesh_features`, delete the lazy `from collab_splats.semantics.lifting import transfer_features` line. Rewrite its docstring as `"""Transfer point features onto mesh vertices on the first mesh query."""`.
8. In `score_query`, change the no-features status to:

```python
            _stage("query: no lifted features for this backend — run the semantics stage")
```

and the "Lift features on first query" comment to `# Read lifted features on first query; none -> plain RGB`.
9. `torch` stays imported (`score_query` uses `torch.from_numpy`).

In `app.py` `_load_outputs.on_done`, rename the kwarg `semantics_dir=semantics_dir` to `lifted_store=semantics_dir`. Task 3 renames the variable.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard -q`
Expected: PASS. `test_score_query_lazy_transfers_mesh_features_when_absent` still passes, because it passes `point_features=` directly.

Run: `/opt/venv/reconstruction/bin/python -m pyflakes collab_splats/dashboard/viewer.py`
Expected: no output.

- [ ] **Step 5: Smoke, then commit**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
git commit --only collab_splats/dashboard/viewer.py collab_splats/dashboard/app.py \
  tests/dashboard/test_viewer.py tests/dashboard/test_viewer_lift.py tests/dashboard/test_app.py \
  -m "refactor(dashboard): viewer reads the lifted store via semantics.store

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Load and query through `Reconstructor`

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Delete: `tests/dashboard/test_semantics_layout.py`
- Modify: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing tests**

Add a fixture helper and new tests to `tests/dashboard/test_app.py`:

```python
BACKEND = "vggt_omega"


def _done_backend(out, backend=BACKEND, extractor=None):
    """Fake a finished pointcloud stage (zarr + COLMAP model) for one backend, optionally its run_config."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / backend / "pointcloud.zarr").mkdir(parents=True)
    (out / backend / "colmap" / "sparse" / "0").mkdir(parents=True)
    if extractor is not None:
        config = {"semantics": {"extractor": extractor}}
        (out / backend / "run_config.yaml").write_text(yaml.safe_dump(config))


def test_reconstructor_reads_the_recorded_run_config(tmp_path):
    """The recorded extractor wins, so outputs['semantics'] names the store that run wrote."""
    app, _ = _app(tmp_path)
    _done_backend(tmp_path / SCENE, extractor="dinov2")
    rec = app._reconstructor(SCENE, BACKEND)
    assert rec.outputs["semantics"] == tmp_path / SCENE / BACKEND / "semantics" / "dinov2_lifted.zarr"
    assert rec.config["output_path"] == str(tmp_path / SCENE)
    assert rec.done("pointcloud")


def test_reconstructor_picks_the_method_from_the_registry(tmp_path):
    app, _ = _app(tmp_path)
    assert app._reconstructor(SCENE, "instantsfm").config["pointcloud"]["method"] == "sfm"
    assert app._reconstructor(SCENE, "vggt_omega").config["pointcloud"]["method"] == "feedforward"


def test_backend_dropdown_lists_every_registered_backend(tmp_path):
    app, _ = _app(tmp_path)
    assert {"vggt_omega", "instantsfm"} <= set(app.backend.options)


def test_load_job_returns_none_for_a_backend_not_run(tmp_path):
    """No pointcloud for this backend locally or remotely -> None; on_done shows the hint."""
    app, worker = _recording_app(tmp_path)
    app._source.has_processed.return_value = False
    app._load_outputs(SCENE, BACKEND)
    job_fn, on_done, _doc = worker.submitted[-1]
    assert job_fn() is None
    on_done(None)
    app._viewer.load.assert_not_called()
    assert any(f"{BACKEND} not run for {SCENE}" in line for line in app._op_log.log_lines)


def test_load_job_reads_a_done_backend(tmp_path, monkeypatch):
    """A done backend loads its result, mesh path and lifted store, cached under (scene, backend)."""
    from collab_splats.pointcloud.base import PointcloudResult

    cache = SceneCache()
    app, worker = _recording_app(tmp_path, cache=cache)
    out = tmp_path / SCENE
    _done_backend(out)
    lifted = out / BACKEND / "semantics" / "talk2dino_lifted.zarr"
    lifted.mkdir(parents=True)
    monkeypatch.setattr(PointcloudResult, "load_zarr", lambda p, **kwargs: "result")
    app._load_outputs(SCENE, BACKEND)
    job_fn, _on_done, _doc = worker.submitted[-1]
    value = job_fn()
    assert value == ("result", None, lifted, "talk2dino")
    assert cache.get((SCENE, BACKEND), "loaded") is value
    app._source.pull_processed.assert_not_called()


def test_switching_backend_reloads(tmp_path, monkeypatch):
    """Changing the backend dropdown loads (scene, new backend)."""
    app, _src = _app(tmp_path)
    _select(app, SCENE)
    loads = []
    monkeypatch.setattr(app, "_load_outputs", lambda scene, backend: loads.append((scene, backend)))
    app.backend.value = "instantsfm"
    assert loads == [(SCENE, "instantsfm")]
```

Adapt the existing tests to `(scene, backend)`:
- Every `app._load_outputs(SCENE)` becomes `app._load_outputs(SCENE, BACKEND)`.
- Every `(tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)` becomes `_done_backend(tmp_path / SCENE)`.
- Every `app._current_scene = SCENE` becomes `app._current_key = (SCENE, BACKEND)`, and every `app._current_scene is None` becomes `app._current_key is None`.
- Cache keys: `cache.put(SCENE, ...)` / `cache.get(SCENE, ...)` become `(SCENE, BACKEND)`, and `app._invalidate_scene(SCENE)` becomes `app._invalidate_scene((SCENE, BACKEND))`.
- `test_load_outputs_on_done_renders_into_viewer`: sentinel `("result", None, "semdir", "talk2dino")`; assert `kwargs["lifted_store"] == "semdir"`; drop the `point_features` assertion.
- `test_load_job_returns_cached_value_without_pull`: sentinel `("result", None, "semdir", "talk2dino")`.
- `test_load_outputs_pull_excludes_dense_arrays`, `test_load_job_reports_pull_progress_to_op_log`, `test_load_outputs_logs_steps`: set `app._source.has_processed.return_value = True` (the job pulls only when the remote has the scene). The fake pull calls `_done_backend(out)` instead of mkdir-ing `pointcloud.zarr`.
- `test_load_does_not_eager_load_features`: use `_done_backend(out)` and create the lifted dir at `out / BACKEND / "semantics" / "talk2dino_lifted.zarr"`.
- `test_autoload_current_noop_on_blank_selection`: delete the `_update_max_frames_bound` monkeypatch line; the `_load_outputs` lambda becomes `lambda scene, backend: probed.append(scene)`.
- `test_set_busy_disables_all_mutating_widgets`: add `app.backend` to the widget tuple.
- `test_view_mode_switch_rescores_active_query_on_worker`: unchanged.

Delete:
- `_write_features_zarr`, `_run_load_job`, `test_load_outputs_resolves_the_flat_semantics_layout`, `test_load_outputs_semantics_is_none_when_scene_has_none`
- every `test_ensure_lift_inputs_*`, every `test_cleanup_lift_inputs_*`, and `_write_cached_features`
- `test_stale_video_meta_apply_skipped`, `test_min_disparity_visibility_tracks_sampling`
- `tests/dashboard/test_semantics_layout.py`: `git rm -q tests/dashboard/test_semantics_layout.py`

- [ ] **Step 2: Run them and confirm they fail**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q`
Expected: FAIL. `AttributeError: 'SplatsApp' object has no attribute '_reconstructor'` / `'backend'`, and `TypeError: _load_outputs() takes 2 positional arguments`.

- [ ] **Step 3: Implement in `app.py`**

1. Imports. Add these to the groups (`mergedeep` is third-party; isort places it):

```python
from mergedeep import merge

from collab_splats.pointcloud import BaseFeedforwardCreator
from collab_splats.pointcloud.sfm import SFM_CREATORS
from collab_splats.reconstructor import Reconstructor
```

Delete the `# NB: collab_splats.dashboard.pipeline and pointcloud.feedforward ...` comment run (the serve warm thread covers import cost).

2. Delete `_scan_output_dirs` and its `# Helpers (kept for backward compat ...)` divider. Delete `_ENV_MODELS`, `_EXTRACTORS`, `_LIFT_MEMBERS`, `_ensure_lift_inputs`, `_cleanup_lift_inputs`, `_update_max_frames_bound`, `_dispatch_load` (no callers: confirm with `rtk proxy grep -rn _dispatch_load collab_splats tests`). Delete the `self.extractor` widget, its `_persisted` entry and its place in the Semantics card.

3. Rename state: `self._current_scene` → `self._current_key: tuple[str, str] | None = None  # (scene, backend) currently displayed` and `self._loading_scene` → `self._loading_key: tuple[str, str] | None = None  # (scene, backend) whose load is in flight`. Add `self._extractor_name = ""  # extractor of the displayed backend's semantics run`. In `_on_density`, set `self._current_key = None`.

4. Backend widget. In `_build_sidebar`, after `self.scene_select`:

```python
        self.backend = pn.widgets.Select(
            name="Backend",
            options=sorted(BaseFeedforwardCreator._registry) + sorted(SFM_CREATORS),
            value=s.get("backend", "vggt_omega"),
        )
```

Then wire `self.backend.param.watch(self._on_scene, "value")` next to the scene watcher. Add `"backend": self.backend` to `_persisted` after `scene_select`. Put `self.backend` under `"## Source"` after `self.scene_select`. Add `self.backend` to the `_set_busy` widget tuple.

5. Add `_reconstructor` after `_ensure_local_video`:

```python
    def _reconstructor(self, scene: str, backend: str, overrides: dict | None = None) -> Reconstructor:
        """Reconstructor for one scene's backend: recorded run config, then overrides, then local paths."""
        out = self._base_dir / scene
        recorded = Reconstructor.run_config_path(out, backend)
        config = yaml.safe_load(recorded.read_text()) if recorded.exists() else {}
        method = "feedforward" if backend in BaseFeedforwardCreator._registry else "sfm"

        # input_path is only read by preproc; the run swaps in the video when preproc must run
        paths = {"input_path": str(out), "output_path": str(out), "pointcloud": {"method": method, "backend": backend}}
        return Reconstructor(merge({}, config, overrides or {}, paths))
```

6. Replace `_autoload_current`:

```python
    def _autoload_current(self) -> None:
        """Load the selected scene's selected backend; the load job pulls it when only the server has it."""
        scene = self.scene_select.value
        if not scene or self._op_log.is_running:
            return

        self._load_outputs(scene, self.backend.value)
```

In `_on_scene`, delete the two-line "probes the frame bound" comment (it refers to the deleted `_update_max_frames_bound`). Change its docstring to `"""Load (scene, backend) when either dropdown changes (skipped during programmatic churn)."""`.

7. Replace `_invalidate_scene` and `_remember_loaded` signatures:

```python
    def _invalidate_scene(self, key: tuple[str, str]) -> None:
        """Drop cached loads for one (scene, backend) after fresh outputs land."""
        self._cache.drop_scene(key)
        if self._current_key == key:
            self._current_key = None  # ensure the next load is not short-circuited
        self._source.invalidate(("has_processed", key[0]))

    def _remember_loaded(self, key: tuple[str, str]) -> None:
```

(Leave the `_remember_loaded` body unchanged.)

8. Replace `_load_outputs`:

```python
    def _load_outputs(self, scene: str, backend: str) -> None:
        """Enqueue loading one backend's result + mesh path + lifted store; render on the IOLoop."""
        key = (scene, backend)

        # Already displayed and idle, or already loading -> nothing to do
        if self._current_key == key and not self._op_log.is_running:
            return
        if self._loading_key == key:
            return

        self._loading_key = key
        out = self._base_dir / scene
        doc = pn.state.curdoc
        max_points = self.max_display_points.value

        def job():
            cached = self._cache.get(key, "loaded")
            if cached is not None:
                self._op_log.append_line(f"{scene}/{backend}: using in-memory cache")
                return cached

            # Pull from the processed bucket when the backend is not done locally
            rec = self._reconstructor(scene, backend)
            if not rec.done("pointcloud") and self._source.has_processed(scene):
                with self._op_log.step(f"{scene}: pulling from server"):
                    self._source.pull_processed(
                        scene,
                        out,
                        excludes=PULL_EXCLUDES,
                        on_line=self._op_log.rclone_progress("⬇ pulling from server"),
                    )
                rec = self._reconstructor(scene, backend)

            if not rec.done("pointcloud"):
                return None

            with self._op_log.step(f"{scene}/{backend}: reading pointcloud.zarr"):
                result = rec.result

            value = (
                result,
                rec.outputs["mesh"] if rec.done("mesh") else None,
                rec.outputs["semantics"] if rec.done("semantics") else None,
                rec.config["semantics"]["extractor"],
            )
            self._cache.put(key, "loaded", value)
            self._remember_loaded(key)
            return value

        def on_done(res):
            self._loading_key = None
            self._sync_busy()
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return

            if res is None:
                self._op_log.append_line(f"{backend} not run for {scene} — press Run")
                self._op_log.finish_op()
                return

            result, mesh_path, lifted_store, extractor_name = res

            # Label the render step with the displayed point count
            try:
                n_pts = len(result.points)
                n_shown = min(n_pts, max_points) if max_points > 0 else n_pts
                label = f"rendering {n_shown:,} points × 2 panes"
            except Exception:  # test doubles without real arrays
                label = "rendering scene"

            with self._op_log.step(label):
                self._viewer.load(result, mesh_path=mesh_path, lifted_store=lifted_store, max_points=max_points)

            self._extractor_name = extractor_name
            self._current_key = key
            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op(f"loading {scene}/{backend}")
        self._gpu.submit(job, on_done, doc)
```

9. In `_on_view_mode`: `scene = self._current_scene` becomes `key = self._current_key`, and the cache calls use `key`. In the job:
- delete `fetched = self._ensure_lift_inputs(scene)` and the `if fetched: ...cleanup` block
- `self._viewer.score_query(...)` returns directly
- `extractor_name` still comes from the active query tuple

10. In `_on_query`:
- `extractor_name = self.extractor.value` becomes `extractor_name = self._extractor_name`
- delete `scene = ...`
- the job body becomes:

```python
        def job():
            return self._viewer.score_query(
                positive=positive, negative=negative, extractor_name=extractor_name, op_log=self._op_log
            )
```

11. Temporary bridge until Task 4: `_on_run` and `_start_run` still call `self._invalidate_scene(scene)` / `self._load_outputs(scene)`. Change those calls to `self._invalidate_scene((scene, self.backend.value))` and `self._load_outputs(scene, self.backend.value)` so the module stays importable. Task 4 rewrites both methods.

12. Delete `shutil` from the imports only if `_ensure_display` no longer uses it. It does use it (`shutil.which`), so `shutil` stays.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard -q`
Expected: PASS.

Run: `/opt/venv/reconstruction/bin/python -m pyflakes collab_splats/dashboard/app.py`
Expected: no unused imports, apart from those Task 4 deletes (`RunConfig`, `run_off_loop`, if flagged).

- [ ] **Step 5: Smoke, then commit**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
git commit --only collab_splats/dashboard/app.py tests/dashboard/test_app.py tests/dashboard/test_semantics_layout.py \
  -m "refactor(dashboard): load (scene, backend) through Reconstructor

Backend dropdown over both creator registries; the recorded run_config picks the lifted store.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Run through `Reconstructor`; delete the dashboard pipeline copy

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Modify: `collab_splats/dashboard/pipeline.py` (down to `_push_async`)
- Modify: `collab_splats/dashboard/serve.py` (`_WARM_MODULES`)
- Delete: `collab_splats/dashboard/config.py`, `tests/dashboard/test_config.py`, `tests/dashboard/test_pipeline.py`
- Modify: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write the failing tests**

Change the `test_app.py` import `from collab_splats.dashboard.config import PULL_EXCLUDES` to `from collab_splats.dashboard.app import PULL_EXCLUDES, SceneCache, SplatsApp` (merge with the existing app import).

Add:

```python
def test_run_job_writes_run_config_and_runs_the_stages(tmp_path, monkeypatch):
    """Run = CLI's three steps: build the Reconstructor, dump its config, run the chosen stages."""
    from collab_splats.reconstructor import Reconstructor

    app, source = _app(tmp_path)
    _select(app, SCENE)
    (tmp_path / SCENE / "images").mkdir(parents=True)  # preproc done -> no video fetch
    app.backend.value = "instantsfm"
    app.stages.value = ["pointcloud"]
    app.overrides.value = "semantics:\n  enabled: false\n"
    calls = []
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages, overwrite: calls.append((stages, overwrite)))
    monkeypatch.setattr("collab_splats.dashboard.app._push_async", lambda *a: None)
    app._on_run(event=None, force=True)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    job_fn()
    assert calls == [(["pointcloud"], True)]
    written = yaml.safe_load(Reconstructor.run_config_path(tmp_path / SCENE, "instantsfm").read_text())
    assert written["pointcloud"]["method"] == "sfm"
    assert written["output_path"] == str(tmp_path / SCENE)
    assert written["semantics"]["enabled"] is False
    source.scene_video.assert_not_called()


def test_run_fetches_the_video_when_preproc_is_not_done(tmp_path, monkeypatch):
    from collab_splats.reconstructor import Reconstructor

    app, _source = _app(tmp_path)
    _select(app, SCENE)
    video = tmp_path / "clip_03.mp4"
    monkeypatch.setattr(app, "_ensure_local_video", lambda scene: video)
    seen = []
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages, overwrite: seen.append(self.config["input_path"]))
    monkeypatch.setattr("collab_splats.dashboard.app._push_async", lambda *a: None)
    app._on_run(event=None, force=False)
    app._gpu.submitted[-1][0]()
    assert seen == [str(video)]


def test_run_rejects_overrides_that_are_not_a_mapping(tmp_path):
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    app.overrides.value = "- just\n- a list\n"
    app._on_run(event=None, force=False)
    assert app._gpu.submitted == []
    assert any("overrides must be a YAML mapping" in line for line in app._op_log.log_lines)


def test_run_on_done_suggests_force_when_output_exists(tmp_path):
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    app._on_run(event=None, force=False)
    _job, on_done, _doc = app._gpu.submitted[-1]
    on_done(ValueError("stage 'pointcloud' output already exists; pass overwrite=True to replace it"))
    assert any("use Force re-run" in line for line in app._op_log.log_lines)


def test_run_on_done_reloads_the_backend(tmp_path, monkeypatch):
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    loads = []
    monkeypatch.setattr(app, "_load_outputs", lambda scene, backend: loads.append((scene, backend)))
    app._on_run(event=None, force=False)
    _job, on_done, _doc = app._gpu.submitted[-1]
    on_done(None)
    assert loads == [(SCENE, app.backend.value)]
```

Adapt / delete in `test_app.py`:
- `test_run_button_submits_pipeline_job`, `test_force_rerun_submits_even_when_cached`: add `_select(app, SCENE)` in place of `app.scene_select.value = SCENE`. In the force test, replace the `pointcloud.zarr` mkdir with `_done_backend(tmp_path / SCENE)`.
- Delete `test_run_loads_cache_without_recompute`, `test_on_run_remote_check_runs_off_loop`, `test_double_click_during_check_fires_single_run`.
- `test_pull_excludes_cover_the_dense_per_pixel_arrays`: unchanged, apart from the import.

Delete the test files: `git rm -q tests/dashboard/test_config.py tests/dashboard/test_pipeline.py`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -q`
Expected: collection error, `ImportError: cannot import name 'PULL_EXCLUDES' from 'collab_splats.dashboard.app'`.

- [ ] **Step 3: Implement in `app.py`**

1. Imports:
- Replace `from collab_splats.dashboard.config import PULL_EXCLUDES, RunConfig` with `from collab_splats.dashboard.pipeline import _push_async`.
- Add `from collab_splats.reconstructor import STAGES, Reconstructor` (merge with the Task 3 import).
- Drop `run_off_loop` from the `async_utils` import if no caller is left. `_autoload_current` no longer uses it; check with `rtk proxy grep -n run_off_loop collab_splats/dashboard/app.py`.

2. Paste `PULL_EXCLUDES` verbatim from `config.py` above `_LOADED_CACHE_KEEP`:

```python
# Dense per-pixel zarr members the viewer never reads; unanchored, so they match under every <backend>/
PULL_EXCLUDES = (
    "pointcloud.zarr/depth/**",
    "pointcloud.zarr/world_points/**",
    "pointcloud.zarr/confidence/**",
    "pointcloud.zarr/features/**",
    "pointcloud.zarr/pixel_indices/**",
    "pointcloud.zarr/images/**",
)
```

3. Delete `_SAMPLERS`, `_MODEL_CONF_DEFAULTS`, `_bind_visibility`, `_on_env_model`, `_current_config`, and the `_cache_check_token` / `_cache_check_active` attributes.
- In `_sync_busy`, `busy = bool(self._gpu.busy)`; delete its two cache-check comment lines.
- Delete the widgets `sampling`, `fps`, `max_frames`, `min_disparity`, `env_model`, `conf`, `mesh_voxel`, `mesh_depth`, together with their `_persisted` entries, their `_bind_visibility` calls, the `env_model` watcher, and their sidebar cards.

4. Add the run widgets after `self.backend` in `_build_sidebar`:

```python
        self.stages = pn.widgets.MultiChoice(
            name="Stages", options=list(STAGES), value=s.get("stages", ["preproc", "pointcloud"])
        )
        self.overrides = pn.widgets.TextAreaInput(
            name="Config overrides (YAML)", placeholder="semantics:\n  enabled: false", value=s.get("overrides", "")
        )
```

Then:
- `_persisted` becomes `scene_select, backend, stages, overrides, pos_query, neg_query, max_display_points, normalize_view`.
- Add `self.stages` and `self.overrides` to the `_set_busy` tuple.
- The sidebar becomes:

```python
        self._sidebar = pn.Column(
            "## Source",
            self.scene_select,
            self.backend,
            pn.Card(self.stages, self.overrides, title="Run", collapsed=False),
            pn.Card(self.pos_query, self.neg_query, self.run_query_btn, title="Semantics", collapsed=False),
            self.max_display_points,
            "### View",
            self.view_mode,
            self.normalize_view,
            pn.Row(self.run_btn, self.force_btn),
            self.busy_note,
        )
```

5. Replace `_on_run` and `_start_run`:

```python
    def _on_run(self, event, force: bool) -> None:
        """Run the chosen stages for (scene, backend); Force re-run overwrites finished stages."""
        self._flush_state()
        scene = self.scene_select.value
        if not scene:
            self._op_log.error_op("select a scene first")
            return

        # Overrides box: a YAML mapping merged over the recorded run config
        try:
            overrides = yaml.safe_load(self.overrides.value) or {}
        except yaml.YAMLError as exc:
            self._op_log.error_op(f"overrides are not valid YAML: {exc}")
            return
        if not isinstance(overrides, dict):
            self._op_log.error_op("overrides must be a YAML mapping")
            return

        self._start_run(scene, self.backend.value, list(self.stages.value) or None, overrides, overwrite=force)

    def _start_run(self, scene: str, backend: str, stages: list[str] | None, overrides: dict, overwrite: bool) -> None:
        """Enqueue a Reconstructor run on the GPU worker, then reload the backend."""
        doc = pn.state.curdoc
        other_backends = [b for b in self.backend.options if b != backend]

        def job():
            rec = self._reconstructor(scene, backend, overrides)

            # Pull the server's backend in full only when it is not done locally, then rebuild from its run_config
            if not rec.done("pointcloud") and self._source.check_available() and self._source.has_processed(scene):
                with self._op_log.step(f"{scene}: pulling from server"):
                    self._source.pull_processed(
                        scene,
                        self._base_dir / scene,
                        excludes=tuple(f"/{b}/**" for b in other_backends),
                        on_line=self._op_log.rclone_progress("⬇ pulling from server"),
                    )
                rec = self._reconstructor(scene, backend, overrides)

            # Fetch the video only when preproc will run; it reads input_path then
            runs_preproc = (stages is None or "preproc" in stages) and (overwrite or not rec.done("preproc"))
            if runs_preproc:
                rec.config["input_path"] = str(self._ensure_local_video(scene))

            # Record the config beside the outputs, as the CLI does
            run_config = Reconstructor.run_config_path(rec.config["output_path"], backend)
            run_config.parent.mkdir(parents=True, exist_ok=True)
            with open(run_config, "w") as f:
                yaml.dump(rec.config, f, default_flow_style=False, sort_keys=False)

            with self._op_log.attach_logging("collab_splats"):
                rec.run(stages, overwrite=overwrite)

            _push_async(self._source, self._base_dir / scene, scene, self._op_log)
            return True

        def on_done(res):
            self._sync_busy()
            if isinstance(res, Exception):
                hint = " — use Force re-run" if "already exists" in str(res) else ""
                self._op_log.error_op(f"{res}{hint}")
                return

            self._invalidate_scene((scene, backend))
            self._load_outputs(scene, backend)

        self._set_busy(True)
        self._op_log.start_op(f"running {scene}/{backend}")
        self._gpu.submit(job, on_done, doc)
```

**Erratum (T4 review):** the job first gated the video fetch on `not rec.done("preproc")`. A Force re-run of a done preproc then rmtree'd `images/` and read the scene dir. The gate is now "preproc will run". When the backend's pointcloud is not done locally, the job also pulls that backend's server outputs, without the dense-array excludes, and then rebuilds the `Reconstructor` so it uses the pulled run_config.yaml. A backend that is done locally is never pulled, so stale server files cannot overwrite newer local outputs.

Leaf runs after a Load pull are unsupported: Load excludes the dense zarr members, so such a run can fail on a missing array.

`OperationLog.attach_logging(*logger_names)` exists (`operation_log.py:118`). It bridges the `collab_splats` loggers into the console for the length of the run.

6. Cut `pipeline.py` down to the module docstring, the imports `_push_async` needs (`logging`, `threading`, `time`, `Path`, `OperationLog`, `SceneSource`), `logger`, and `_push_async` verbatim. The docstring becomes `"""Detached push of a scene's outputs to the processed bucket."""`. Verify with `/opt/venv/reconstruction/bin/python -m pyflakes collab_splats/dashboard/pipeline.py`; expect no output.

7. `git rm -q collab_splats/dashboard/config.py`.

8. In `serve.py` `_WARM_MODULES`, replace the `dashboard.pipeline` entry with `("collab_splats.reconstructor", "reconstruction pipeline")`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard -q`
Expected: PASS.

Run: `rtk proxy grep -rn "RunConfig\|run_pipeline\|resolve_semantics_dir\|dashboard.config\|_cache_check\|env_model\|max_frames" collab_splats/dashboard tests/dashboard`
Expected: no hits.

Run: `/opt/venv/reconstruction/bin/python -m pyflakes collab_splats/dashboard`
Expected: no unused imports or undefined names.

- [ ] **Step 5: Smoke, then commit**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
git commit --only collab_splats/dashboard tests/dashboard \
  -m "refactor(dashboard): run through Reconstructor; delete the dashboard pipeline copy

Stages + YAML overrides widgets; config.py and run_pipeline go; pipeline.py keeps _push_async.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Gates and acceptance

**Files:** none modified (unless a gate fails)

- [ ] **Step 1: Containment**

Run: `cd $WT && git diff --name-only rebase/ocr-lens..HEAD`
Expected: only paths under `collab_splats/dashboard/`, `tests/dashboard/`, `docs/superpowers/`.

- [ ] **Step 2: Formatting and import style**

```bash
cd $WT && /opt/venv/reconstruction/bin/python -m isort --check-only collab_splats/dashboard tests/dashboard
/opt/venv/reconstruction/bin/python -m black --check collab_splats/dashboard tests/dashboard
PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/test_import_style.py -q
```

Expected: all clean. If isort or black flag a file, run them on those files only (never repo-wide), re-run the gates, and commit with `style(dashboard): ...`.

- [ ] **Step 3: Dashboard suite and smoke**

```bash
PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard -q
PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: PASS and exit 0. Report the test count next to the Task 0 baseline.

- [ ] **Step 4: Full suite (no other heavy jobs running)**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests -q -x --timeout 1800` inside tmux.
Expected: 0 failures outside `docs/known-test-failures.md`. Any new failure outside `tests/dashboard` means containment broke; investigate before going on.

- [ ] **Step 5: Manual acceptance (human, on `/workspace/outputs/2026_07_15-Goprosplat-GH010229`)**

1. Serve: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --base-dir /workspace/outputs`
2. Pick the scene with backend `vggt_omega` → the point cloud renders.
3. Backend `instantsfm`, stages `pointcloud`, overrides `semantics: {enabled: false}` → Run. The run uses the existing `images/` and does not fetch the video.
4. Switch the dropdown between the two backends → both render.
5. On one backend, run the `semantics` stage, then query "chair" → the right pane recolors.
6. Run `pointcloud` again without Force → the error ends with "— use Force re-run".

- [ ] **Step 6: Changelog (after the user accepts)**

Append a `dashboard-release` entry to `docs/superpowers/CHANGELOG.md`, and move the CLAUDE.md In-Flight line to Recently Completed. Commit with `git add -f` + `git commit --only`.
