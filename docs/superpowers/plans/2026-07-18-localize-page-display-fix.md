# Localize Page Display Fix + DB Browse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Localize page actually render (frame preview + run results), and show an existing DB's localized cameras/thumbnails on scene select without a run.

**Architecture:** Split display state from panes in `LocalizePage`: state (last preview frame / run output / browse data) lives on the page; all result panes are constructed fresh inside `main()` per document build and painted from state. A new non-GPU browse loader in `pipeline.py` reads stored localized poses from the zarr `localized/` group.

**Tech Stack:** Panel/Bokeh, pyvista/VTK, matplotlib (Agg), zarr v3 (`compressors=[BloscCodec(...)]`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-18-localize-page-display-fix-design.md`

**Environment:** Always use `/opt/venv/reconstruction/bin/python` (py3.11). Test command: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -x -q`. Pre-commit gate (Task 6): `python -m collab_splats.dashboard --smoke` must print SMOKE PASS.

**Code style (repo rules):** imports at top (exception: heavy lazy imports inside run/render paths, already the pattern in these files); block-level comments; one-line docstrings on public functions; flat test functions; `logging` not `print`.

---

## Background for the implementing engineer

- `collab_splats/dashboard/localize.py` — the page being fixed. Read it fully first.
- `collab_splats/dashboard/shell.py` — `DashboardShell.view()` builds tabs; `LocalizePage.main()` is called lazily on first Localize-tab activation (`_on_tab`). Sidebar widgets exist from `__init__`.
- **The bug:** result panes (`_frame_pane`, `_matches_col`, `_dist_pane`, `_stats`, VTK) are built in `__init__` (no server document yet) and mutated later; panes bound to a stale/absent Bokeh doc silently drop updates. Also `_matches_col` (stretch_width) can flex-collapse beside the stretch_both VTK pane.
- **Zarr layout** (written by `CameraLocalizer._append_localized_to_zarr`, `collab_splats/localization/localizer.py`): group `local_features/<extractor>/localized` holds arrays `extrinsics` (L,4,4 float32 world-to-camera), `keypoints`, `descriptors`, `frame_offsets`, optional `scores`; group attrs `image_paths` (list of absolute path strings from the *building* machine — only basenames are portable) and `provenance` (list of dicts). The group is absent until the first localized frame is appended.
- `FeedforwardResult.extrinsics` (`collab_splats/pointcloud/feedforward/base.py:57`) is (N,4,4) float32 world-to-camera homogeneous — use directly, no conversion.
- `pipeline.py` already imports `zarr`, has `PULL_EXCLUDES`, `_load_feedforward_result(out_dir)` (minimal-member zarr load).
- Existing tests: `tests/dashboard/test_localize_page.py` — flat functions, `_page(tmp_path)` helper builds a `LocalizePage` with `MagicMock` source/worker. `tests/dashboard/test_pipeline.py` exists for pipeline helpers.
- Tests must NOT construct a real `pv.Plotter` (offscreen GL may be absent in CI). The restructure keeps VTK creation out of `_build_panes()`'s hard path: the VTK pane is created with `self._plotter.ren_win if self._plotter is not None else None`, so tests call `_build_panes()`/`main()`-adjacent paths without ever calling `_ensure_plotter()`.

---

### Task 1: Browse data loader in pipeline.py

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py` (add `BrowseData`, `read_localized_group`, `load_browse_data` — place after `_local_ref_paths`, before `run_localization`)
- Test: `tests/dashboard/test_pipeline.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_pipeline.py` (match its existing imports; add the ones below if missing):

```python
import numpy as np
import zarr
from zarr.codecs import BloscCodec

from collab_splats.dashboard.pipeline import read_localized_group


def _make_localized_zarr(tmp_path, extractor="loma", n=2):
    """Minimal feedforward.zarr with a localized/ group for one extractor."""
    zpath = tmp_path / "feedforward.zarr"
    store = zarr.open(str(zpath), mode="a")
    group = store.require_group(f"local_features/{extractor}/localized")
    lz4 = BloscCodec(cname="lz4")
    ext = np.stack([np.eye(4, dtype=np.float32) * (i + 1) for i in range(n)])
    group.create_array("extrinsics", data=ext, chunks=(1, 4, 4), compressors=lz4)
    group.attrs["image_paths"] = [f"/builder/machine/localized_frames/f{i}.jpg" for i in range(n)]
    return zpath, ext


def test_read_localized_group_returns_poses_and_local_paths(tmp_path):
    zpath, ext = _make_localized_zarr(tmp_path, extractor="loma", n=2)
    poses, paths = read_localized_group(zpath, "loma", tmp_path)
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses, ext)
    # Paths remapped to this machine's localized_frames/ by basename
    assert paths == [tmp_path / "localized_frames" / "f0.jpg", tmp_path / "localized_frames" / "f1.jpg"]


def test_read_localized_group_missing_group_is_empty(tmp_path):
    zpath = tmp_path / "feedforward.zarr"
    zarr.open(str(zpath), mode="a")  # store exists, no localized group
    poses, paths = read_localized_group(zpath, "disk", tmp_path)
    assert poses.shape == (0, 4, 4)
    assert paths == []


def test_load_browse_data_composes_result_and_zarr(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from collab_splats.dashboard.pipeline import load_browse_data
    from collab_splats.dashboard.operation_log import OperationLog

    out_dir = tmp_path / "sess" / "vid"
    out_dir.mkdir(parents=True)
    zpath, _ = _make_localized_zarr(out_dir, extractor="loma", n=1)
    ref_ext = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    monkeypatch.setattr(
        "collab_splats.dashboard.pipeline._load_feedforward_result",
        lambda d: SimpleNamespace(extrinsics=ref_ext),
    )
    source = MagicMock()
    data = load_browse_data(
        session="sess", stem="vid", extractor="loma",
        source=source, base_dir=tmp_path, op_log=OperationLog(),
    )
    source.pull_processed.assert_not_called()  # zarr already local -> no pull
    assert data.extractor == "loma"
    assert data.ref_extrinsics.shape == (3, 4, 4)
    assert data.localized_extrinsics.shape == (1, 4, 4)
    assert data.mesh_path == out_dir / "mesh" / "mesh_tsdf.ply"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_pipeline.py -q -k "localized_group or browse_data"`
Expected: FAIL — `ImportError: cannot import name 'read_localized_group'`

- [ ] **Step 3: Implement in pipeline.py**

Add after `_local_ref_paths` (uses existing module imports `zarr`, `np`, `Path`, `dataclass`; `PULL_EXCLUDES` and `_load_feedforward_result` already exist in the module):

```python
@dataclass
class BrowseData:
    """Stored-DB view of a scene: reconstruction cameras + previously localized poses."""

    extractor: str
    ref_extrinsics: np.ndarray  # (N, 4, 4) world-to-camera reconstruction cameras
    localized_extrinsics: np.ndarray  # (L, 4, 4) stored localized poses (L may be 0)
    localized_image_paths: list  # local localized_frames/ paths (existence not guaranteed)
    mesh_path: Path  # scene mesh (may not exist)


def read_localized_group(zarr_path: Path, extractor: str, out_dir: Path) -> "tuple[np.ndarray, list]":
    """Read stored localized poses + local image paths for one extractor (read-only, no GPU)."""
    store = zarr.open(str(zarr_path), mode="r")
    key = f"local_features/{extractor}/localized"
    if key not in store:
        return np.zeros((0, 4, 4), dtype=np.float32), []
    group = store[key]
    poses = np.asarray(group["extrinsics"])
    # image_paths attrs were recorded on the building machine — only basenames are portable
    names = [Path(p).name for p in group.attrs.get("image_paths", [])]
    return poses, [Path(out_dir) / "localized_frames" / n for n in names]


def load_browse_data(
    *,
    session: str,
    stem: str,
    extractor: str,
    source: SessionSource,
    base_dir: Path,
    op_log: OperationLog,
) -> BrowseData:
    """Non-GPU DB browse load: minimal pull if absent, then ref extrinsics + stored localized poses."""
    out_dir = Path(base_dir) / session / stem
    # Minimal pull only when the zarr is not yet local (same excludes as run_localization)
    if not (out_dir / "feedforward.zarr").exists():
        with op_log.step("browse: pulling reconstruction"):
            source.pull_processed(session, stem, out_dir, excludes=PULL_EXCLUDES)
    result = _load_feedforward_result(out_dir)
    loc_ext, loc_paths = read_localized_group(out_dir / "feedforward.zarr", extractor, out_dir)
    return BrowseData(
        extractor=extractor,
        ref_extrinsics=np.asarray(result.extrinsics),
        localized_extrinsics=loc_ext,
        localized_image_paths=loc_paths,
        mesh_path=out_dir / "mesh" / "mesh_tsdf.ply",
    )
```

Note: if `SessionSource`/`OperationLog` are not already imported at the top of `pipeline.py`, use string annotations instead of adding heavy imports (check the file's existing import block first — `OperationLog` is already imported there).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_pipeline.py -q`
Expected: PASS (all, including pre-existing)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/pipeline.py tests/dashboard/test_pipeline.py
git commit -m "feat(dashboard): non-GPU browse loader for stored localization DBs"
```

---

### Task 2: Display-state restructure in localize.py

**Files:**
- Modify: `collab_splats/dashboard/localize.py`
- Test: `tests/dashboard/test_localize_page.py`

This is the core fix. Replace the `__init__`-built result panes with page state + per-build panes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_localize_page.py`:

```python
def test_build_panes_returns_fresh_objects_each_call(tmp_path):
    """Panes must be per-document: two builds share no pane objects (stale-doc bug class)."""
    page = _page(tmp_path)
    a = page._build_panes()
    b = page._build_panes()
    assert set(a) == {"matches_col", "vtk", "dist", "stats"}
    assert all(a[k] is not b[k] for k in a)


def test_show_frame_before_panes_is_pending_then_renders(tmp_path):
    """_show_frame before main() must not crash; the frame renders when panes build."""
    page = _page(tmp_path)
    assert page._panes is None
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    page._show_frame(frame)  # no panes yet -> state only
    assert page._state["left"] == "frame"
    page._panes = page._build_panes()
    page._render_state()
    import panel as pn

    assert isinstance(page._panes["matches_col"][0], pn.pane.Image)


def test_show_frame_downscales_to_thumbnail_v2(tmp_path):
    """Preview render caps thumbnail width at _PREVIEW_MAX_W."""
    page = _page(tmp_path)
    page._panes = page._build_panes()
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    page._show_frame(big)
    pane = page._panes["matches_col"][0]
    assert pane.object.width <= 640


def test_render_state_run_precedence(tmp_path, monkeypatch):
    """left='run' paints figures + stats and draws the scene."""
    import matplotlib.figure
    from types import SimpleNamespace

    page = _page(tmp_path)
    page._panes = page._build_panes()
    drawn = []
    monkeypatch.setattr(page, "_render_scene", lambda *a, **k: drawn.append(a))
    loc = SimpleNamespace(pose=np.eye(4, dtype=np.float32))
    out = SimpleNamespace(result=loc, ref_extrinsics=np.eye(4, dtype=np.float32)[None])
    figs = {
        "dist_fig": matplotlib.figure.Figure(),
        "match_figs": [matplotlib.figure.Figure()],
        "stats_html": "<div>stats</div>",
    }
    page._state["run"] = (out, figs, None)
    page._state["left"] = "run"
    page._render_state()
    assert page._panes["dist"].object is figs["dist_fig"]
    assert page._panes["stats"].object == "<div>stats</div>"
    assert len(drawn) == 1
```

Also **delete** the old `test_show_frame_downscales_to_thumbnail` (it asserts the removed `page._frame_pane` attribute).

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -q -k "build_panes or pending or thumbnail_v2 or run_precedence"`
Expected: FAIL — `AttributeError: 'LocalizePage' object has no attribute '_build_panes'`

- [ ] **Step 3: Restructure localize.py**

3a. Add a module constant next to `_PREVIEW_MAX_W`:

```python
_LEFT_W = _PREVIEW_MAX_W + 24  # fixed left-column width: flex beside the VTK pane collapses to zero
```

3b. Replace `_build_main` entirely with:

```python
    def _build_main(self) -> None:
        """Display STATE only — panes are built per document in main() (_build_panes).

        Panes constructed before a server document exists (the old __init__ pattern) bind
        to a stale/absent Bokeh doc and silently drop updates; nothing display-bound may
        outlive a main() build. Watchers firing before main() (e.g. _show_frame via
        _on_query_video during __init__) write state here and render on build.
        """
        # left: which content owns the left column — 'placeholder' | 'frame' | 'run' | 'browse'
        self._state: dict = {"frame": None, "run": None, "browse": None, "left": "placeholder"}
        self._panes: "dict | None" = None
        self._plotter: pv.Plotter | None = None
```

3c. Keep `_ensure_plotter` as-is but REMOVE its `self._vtk_pane` line (the pane is per-build now):

```python
    def _ensure_plotter(self) -> None:
        """Build the off-screen pyvista plotter on first use (lazy: main/_render_scene)."""
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=True)
```

3d. Add `_build_panes` + `_render_state` + `_render_preview` (new methods, after `_ensure_plotter`):

```python
    def _build_panes(self) -> dict:
        """Construct fresh result panes for the current document build.

        VTK pane object is None when no plotter exists (tests never create one)."""
        ren = self._plotter.ren_win if self._plotter is not None else None
        return {
            "matches_col": pn.Column(width=_LEFT_W, scroll=True, max_height=700),
            "vtk": pn.pane.VTK(ren, sizing_mode="stretch_both", min_height=500),
            "dist": pn.pane.Matplotlib(None, sizing_mode="stretch_width", tight=True),
            "stats": pn.pane.HTML("", sizing_mode="stretch_width"),
        }

    def _render_state(self) -> None:
        """Paint the current panes from page state (called by main() and every event render)."""
        if self._panes is None:
            return
        left = self._state["left"]
        if left == "run" and self._state["run"] is not None:
            out, figs, mesh = self._state["run"]
            self._render_result(out, figs, mesh)
        elif left == "browse" and self._state["browse"] is not None:
            self._render_browse(self._state["browse"])
        elif left == "frame" and self._state["frame"] is not None:
            self._render_preview(self._state["frame"])
        else:
            self._panes["matches_col"][:] = [
                pn.pane.HTML(
                    "<i style='color:#888'>Select a scene and a query video, then Run. "
                    "The selected frame previews here; progress shows in the Operations console.</i>"
                )
            ]

    def _render_preview(self, frame: np.ndarray) -> None:
        """Show the selected query frame in the left panel, downscaled to a thumbnail."""
        from PIL import Image as PILImage

        # Full-res frames push MBs of base64 into the doc — cap the preview width
        img = PILImage.fromarray(frame)
        if img.width > _PREVIEW_MAX_W:
            img = img.resize((_PREVIEW_MAX_W, max(1, int(img.height * _PREVIEW_MAX_W / img.width))))
        self._panes["matches_col"][:] = [pn.pane.Image(img, width=_PREVIEW_MAX_W)]
```

(`_render_browse` is a stub for now — add `def _render_browse(self, browse) -> None: pass` with docstring `"""Filled in by the DB-browse task."""`; Task 3 replaces it.)

3e. Replace `main()`:

```python
    def main(self) -> pn.Column:
        """Per-document build: fresh panes, painted from page state (last preview/run/browse)."""
        self._ensure_plotter()
        self._panes = self._build_panes()
        self._render_state()

        # Busy-state poll only: the operations console is rendered ONCE by DashboardShell,
        # outside the tabs, so it stays visible on both tabs.
        try:
            pn.state.add_periodic_callback(self._sync_busy, period=300, start=True)
        except Exception:
            logger.debug("no periodic callback (no server doc)", exc_info=True)

        top = pn.Row(self._panes["matches_col"], self._panes["vtk"], sizing_mode="stretch_both")
        bottom = pn.Column(self._panes["dist"], self._panes["stats"], sizing_mode="stretch_width")
        return pn.Column(top, bottom, sizing_mode="stretch_both")
```

3f. Replace `_show_frame`:

```python
    def _show_frame(self, frame: np.ndarray) -> None:
        """Record the selected query frame as page state and render it if panes exist."""
        self._state["frame"] = frame
        self._state["left"] = "frame"
        self._render_state()
```

3g. Add `_set_run_state` and rewire `_on_run.on_done`; `_render_result` becomes pure pane assignment:

```python
    def _set_run_state(self, res) -> None:
        """Swap in a new run result; close the superseded run's figures (pyplot Gcf)."""
        import matplotlib.pyplot as plt

        old = self._state.get("run")
        if old is not None:
            _, old_figs, _ = old
            for f in [old_figs["dist_fig"], *old_figs["match_figs"]]:
                plt.close(f)
        self._state["run"] = res
        self._state["left"] = "run"
```

In `_on_run`, replace `on_done` with:

```python
        def on_done(res):
            # Sync, not unconditional re-enable: another queued job must keep widgets locked.
            self._sync_busy()
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._set_run_state(res)
            self._render_state()
```

Replace `_render_result` with (drop all its plt.close logic — superseded figures are closed in `_set_run_state`):

```python
    def _render_result(self, out, figs: dict, mesh) -> None:
        """Fill all three panels from pre-built figures + mesh (IOLoop thread — assignment only)."""
        if self._panes is None:
            return
        try:
            self._panes["dist"].object = figs["dist_fig"]
            self._panes["stats"].object = figs["stats_html"]

            # Left: top-k match-pair figures, best-first (replaces the frame preview)
            if figs["match_figs"]:
                self._panes["matches_col"][:] = [
                    pn.pane.Matplotlib(f, width=_LEFT_W - 24, tight=True) for f in figs["match_figs"]
                ]
            else:
                self._panes["matches_col"][:] = [pn.pane.HTML("<i>no match figures</i>")]

            # Right: mesh + viridis reconstruction cameras + red localized camera
            pose = out.result.pose
            localized = pose[np.newaxis] if pose is not None else None
            self._render_scene(mesh, out.ref_extrinsics, localized)
        except Exception as exc:
            # Surface a render failure via the op_log instead of escaping to the IOLoop
            logger.warning("localize render failed", exc_info=True)
            self._op_log.error_op(str(exc))
```

3h. Generalize `_render_scene` — `localized` is now a (K,4,4) array or None (run passes one pose; browse will pass all stored poses), and the VTK pane comes from `self._panes`:

```python
    def _render_scene(self, mesh, extrinsics: np.ndarray, localized: "np.ndarray | None") -> None:
        """Rebuild the 3D pane: mesh, time-coloured cameras, red localized camera(s)."""
        self._ensure_plotter()
        self._plotter.clear()

        # Mesh arrives preloaded (worker read it via _ensure_scene_mesh)
        if mesh is not None:
            self._plotter.add_mesh(mesh, rgb="RGB" in mesh.array_names, opacity=0.9)

        # Reconstruction cameras: viridis by time; subsampled with an on-plot note
        centers = camera_centers(np.asarray(extrinsics))
        step = subsample_step(len(centers))
        sub = centers[::step]
        poly = pv.PolyData(sub)
        poly["time"] = np.arange(len(sub), dtype=np.float32)
        self._plotter.add_mesh(
            poly, scalars="time", cmap="viridis", point_size=14, render_points_as_spheres=True, show_scalar_bar=False
        )
        if step > 1:
            self._plotter.add_text(f"showing every {step}rd camera", font_size=8, position="lower_left")

        # Localized camera(s) in red, drawn larger
        if localized is not None and len(localized):
            loc_centers = camera_centers(np.asarray(localized))
            self._plotter.add_mesh(
                pv.PolyData(loc_centers), color="red", point_size=22, render_points_as_spheres=True
            )

        self._plotter.reset_camera()
        if self._panes is not None and self._panes["vtk"].object is not None:
            self._panes["vtk"].synchronize()
```

3i. Remove now-dead attributes everywhere: `self._frame_pane`, `self._matches_col`, `self._dist_pane`, `self._stats`, `self._vtk_pane`. Grep the file for each name after editing — zero hits outside comments.

- [ ] **Step 4: Run localize page tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py tests/dashboard/test_shell.py -q`
Expected: PASS. If a pre-existing test references a removed attribute, update it to go through `_build_panes()` + `_state` (same pattern as the new tests).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/localize.py tests/dashboard/test_localize_page.py
git commit -m "fix(dashboard): per-document pane construction on localize page

Result panes built in __init__ bound to a stale/absent Bokeh doc and
silently dropped updates (blank preview, blank run results). Display
state now lives on the page; main() builds fresh panes per document and
renders state. Fixed-width left column prevents flex-collapse beside
the VTK pane."
```

---

### Task 3: DB browse on scene select

**Files:**
- Modify: `collab_splats/dashboard/localize.py`
- Test: `tests/dashboard/test_localize_page.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_localize_page.py`:

```python
def test_render_browse_paints_header_and_scene(tmp_path, monkeypatch):
    """Browse render: header count in left column, all stored poses passed to the scene."""
    from types import SimpleNamespace

    page = _page(tmp_path)
    page._panes = page._build_panes()
    drawn = {}
    monkeypatch.setattr(
        page, "_render_scene", lambda mesh, ext, localized: drawn.update(mesh=mesh, ext=ext, localized=localized)
    )
    data = SimpleNamespace(
        extractor="loma",
        ref_extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0),
        localized_extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        localized_image_paths=[tmp_path / "missing0.jpg", tmp_path / "missing1.jpg"],
    )
    page._state["browse"] = (data, None)
    page._state["left"] = "browse"
    page._render_state()
    header = page._panes["matches_col"][0]
    assert "2 localized frames" in header.object
    assert drawn["localized"].shape == (2, 4, 4)
    assert page._panes["dist"].object is None  # browse clears stale run figures


def test_scene_video_select_triggers_browse_load(tmp_path, monkeypatch):
    """Choosing a scene video spawns the browse load with the preselected extractor."""
    import time
    from types import SimpleNamespace

    page = _page(tmp_path)
    calls = []
    monkeypatch.setattr(page, "_load_browse", lambda session, stem, extractor, doc: calls.append(extractor))
    page._source.list_localization_dbs.return_value = ["loma"]
    page.scene_session.options = ["sess"]
    page.scene_session.value = "sess"
    page._on_scene_video(SimpleNamespace(new="vid"))
    for _ in range(50):  # _on_scene_video runs its work() on a thread
        if calls:
            break
        time.sleep(0.1)
    assert calls == ["loma"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -q -k "browse"`
Expected: FAIL — browse render is a stub / `_load_browse` missing.

- [ ] **Step 3: Implement browse in localize.py**

3a. Replace the `_render_browse` stub:

```python
    def _render_browse(self, browse) -> None:
        """Paint the stored-DB view: thumbnail strip left, mesh + stored poses right."""
        data, mesh = browse
        n = len(data.localized_extrinsics)
        header = pn.pane.HTML(
            f"<b>DB ({data.extractor})</b>: {n} localized frame{'s' if n != 1 else ''}"
        )
        # Thumbnails only for images present locally (pull may not include every frame)
        thumbs = [
            pn.pane.Image(str(p), width=_PREVIEW_MAX_W // 2)
            for p in data.localized_image_paths
            if Path(p).exists()
        ]
        self._panes["matches_col"][:] = [header, *thumbs]
        # Browse owns the page: clear stale run figures below
        self._panes["dist"].object = None
        self._panes["stats"].object = ""
        self._render_scene(mesh, data.ref_extrinsics, data.localized_extrinsics)
```

3b. Add `_load_browse` (place near `_ensure_scene_mesh`):

```python
    def _load_browse(self, session: str, stem: str, extractor: str, doc) -> None:
        """Background thread: load stored-DB browse data + mesh, then render (no GPU)."""
        try:
            from collab_splats.dashboard.pipeline import load_browse_data

            data = load_browse_data(
                session=session,
                stem=stem,
                extractor=extractor,
                source=self._source,
                base_dir=self._base_dir,
                op_log=self._op_log,
            )
            mesh = self._ensure_scene_mesh((session, stem), data.mesh_path)
        except Exception as exc:
            logger.warning("browse load failed", exc_info=True)
            self._op_log.append_line(f"DB browse FAILED: {exc}")
            return

        def show():
            self._state["browse"] = (data, mesh)
            self._state["left"] = "browse"
            self._render_state()
            self._update_db_note()

        doc.add_next_tick_callback(show) if doc is not None else show()
```

3c. Chain browse into `_on_scene_video.work()` — after the setter dispatch, still on the worker thread (the preselected extractor is computed locally to avoid racing the IOLoop setter):

```python
        def work():
            # step() logs start/done and FAILED; bail on failure (note stays as-is).
            try:
                with self._op_log.step("listing feature DBs"):
                    dbs = self._source.list_localization_dbs(session, stem)
            except Exception as exc:
                logger.warning("feature DB listing failed: %s", exc, exc_info=True)
                return

            def setter():
                options, value = preselect_method(dbs, _METHODS)
                self.method.options = options
                self.method.value = value
                self._dbs = dbs
                self._update_db_note()

            doc.add_next_tick_callback(setter) if doc is not None else setter()

            # Stored-DB browse: render existing localized poses without a run (non-GPU).
            # Extractor computed here, not read from the widget — the setter races us.
            _, extractor = preselect_method(dbs, _METHODS)
            self._load_browse(session, stem, extractor, doc)
```

3d. Re-browse on method change — replace `_on_method`:

```python
    def _on_method(self, event) -> None:
        """Refresh the DB note and re-browse the stored DB for the newly picked extractor."""
        self._update_db_note()
        session, stem = self.scene_session.value, self.scene_video.value
        if not (session and stem) or self._gpu.busy:
            return
        threading.Thread(
            target=self._load_browse,
            args=(session, stem, event.new, pn.state.curdoc),
            name="browse-load",
            daemon=True,
        ).start()
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/localize.py tests/dashboard/test_localize_page.py
git commit -m "feat(dashboard): show stored localization DB on scene select

Selecting a scene video now loads the existing DB (minimal pull if
needed) and renders mesh + reconstruction cameras + stored localized
poses with a thumbnail strip - no GPU run required."
```

---

### Task 4: db_note localized-frame count

**Files:**
- Modify: `collab_splats/dashboard/localize.py` (`_update_db_note`)
- Test: `tests/dashboard/test_localize_page.py`

- [ ] **Step 1: Write the failing test**

```python
def test_db_note_includes_localized_count(tmp_path):
    from types import SimpleNamespace

    page = _page(tmp_path)
    page._dbs = ["loma"]
    page.method.options = ["loma", "disk"]
    page.method.value = "loma"
    data = SimpleNamespace(extractor="loma", localized_extrinsics=np.zeros((3, 4, 4), np.float32))
    page._state["browse"] = (data, None)
    page._update_db_note()
    assert "3 localized frames" in page.db_note.object
    # Count belongs to loma's browse data — a different method must not show it
    page.method.value = "disk"
    assert "localized frames" not in page.db_note.object
```

Note: setting `method.value` fires `_on_method` → `_update_db_note` via the watcher, so no second explicit call is needed after the value flip (scene widgets are empty in `_page`, so no browse thread spawns).

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py::test_db_note_includes_localized_count -q`
Expected: FAIL — count missing from note.

- [ ] **Step 3: Implement**

Replace `_update_db_note`:

```python
    def _update_db_note(self) -> None:
        """DB status for the selected method: reuse (+ localized count) or build-on-run warning."""
        dbs = getattr(self, "_dbs", [])
        browse = self._state.get("browse")
        count = ""
        if browse is not None and browse[0].extractor == self.method.value:
            n = len(browse[0].localized_extrinsics)
            count = f" ({n} localized frame{'s' if n != 1 else ''})"
        if self.method.value in dbs:
            self.db_note.object = f"<span style='color:#50c050;font-size:11px'>DB exists — will reuse{count}</span>"
        else:
            self.db_note.object = (
                "<span style='color:#e0a050;font-size:11px'>no DB for this method — "
                "Run will build it (GPU, minutes)</span>"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/localize.py tests/dashboard/test_localize_page.py
git commit -m "feat(dashboard): localized-frame count in localize DB note"
```

---

### Task 5: Run-appends refresh browse state

**Files:**
- Modify: `collab_splats/dashboard/localize.py` (`_on_run.on_done`)
- Test: `tests/dashboard/test_localize_page.py`

A run with `append_to_db` adds a frame the cached browse state doesn't know about; the stale count would then show in db_note. Invalidate browse state after a successful run so the next browse (scene/method reselect) reloads it.

- [ ] **Step 1: Write the failing test**

```python
def test_run_success_invalidates_browse_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    page = _page(tmp_path)
    page._state["browse"] = (SimpleNamespace(extractor="loma", localized_extrinsics=np.zeros((1, 4, 4))), None)
    monkeypatch.setattr(page, "_render_state", lambda: None)
    figs = {"dist_fig": None, "match_figs": [], "stats_html": ""}
    page._handle_run_done((SimpleNamespace(), figs, None))
    assert page._state["browse"] is None
    assert page._state["left"] == "run"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py::test_run_success_invalidates_browse_state -q`
Expected: FAIL — `_handle_run_done` missing.

- [ ] **Step 3: Implement**

Extract `on_done`'s success path into a testable method and call it from `_on_run`:

```python
    def _handle_run_done(self, res) -> None:
        """Success path of a run: swap run state in, drop stale browse state, render."""
        self._set_run_state(res)
        # A run may have appended to the DB — cached browse data (and its count) is stale
        self._state["browse"] = None
        self._render_state()
```

In `_on_run`, `on_done` becomes:

```python
        def on_done(res):
            # Sync, not unconditional re-enable: another queued job must keep widgets locked.
            self._sync_busy()
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._handle_run_done(res)
```

Guard in `_set_run_state` for `dist_fig=None` (test fixture): change the close loop to

```python
            for f in [old_figs["dist_fig"], *old_figs["match_figs"]]:
                if f is not None:
                    plt.close(f)
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/localize.py tests/dashboard/test_localize_page.py
git commit -m "fix(dashboard): drop stale browse state after a localization run"
```

---

### Task 6: Full gate + format

- [ ] **Step 1: Full dashboard suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -q`
Expected: PASS (compare against `docs/known-test-failures.md` if anything unrelated fails).

- [ ] **Step 2: Full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
Expected: matches known-failure baseline; no new failures.

- [ ] **Step 3: Smoke gate (mandatory before dashboard commits)**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: prints `SMOKE PASS`.

- [ ] **Step 4: Format + lint pass**

Run: `black collab_splats/dashboard/localize.py collab_splats/dashboard/pipeline.py tests/dashboard/ && isort collab_splats/dashboard/localize.py collab_splats/dashboard/pipeline.py tests/dashboard/`
Commit any reformat as `style(dashboard): format localize display fix`.

- [ ] **Step 5: Manual browser checklist (report to user, do not skip silently)**

1. Serve dashboard, open Localize tab.
2. Select field session/camera/query video → frame 0 appears in left column after fetch completes.
3. Scrub slider → preview updates.
4. Select a scene with an existing DB → browse view appears (3D mesh + viridis cameras + red stored poses; thumbnail strip; db_note shows count) without pressing Run.
5. Run → match figures replace left column, inlier distribution + stats fill bottom, red camera in 3D pane.
6. Switch to Splats tab and back → last content still renders.

---

## Self-review notes

- Spec §1 (per-doc panes, pending state, fixed width, re-render restores content) → Task 2. Re-render restore: `main()` calls `_render_state()`, which repaints from state — covered.
- Spec §2 (full download policy) → no code change; preview appearance guaranteed by Task 2.
- Spec §3 (browse on scene select, pull-if-missing, 3D + thumbnails, SceneCache) → Tasks 1 + 3. Mesh cached under existing `"mesh"` kind via `_ensure_scene_mesh`; browse data itself is page state (per-scene reload is a cheap local zarr read after first pull — no new cache kind needed; deviation from spec's "new browse kind" noted as simplification).
- Spec §4 (db_note count) → Task 4; staleness after append handled by Task 5.
- Spec §5 (tests, smoke gate, manual checklist) → per-task tests + Task 6.
- Type consistency: `_render_scene(mesh, extrinsics, localized: (K,4,4)|None)` used identically in Tasks 2 and 3; `_state` keys `frame|run|browse|left` consistent throughout; `BrowseData` fields match between Task 1 (definition) and Tasks 3–4 (SimpleNamespace stand-ins).
