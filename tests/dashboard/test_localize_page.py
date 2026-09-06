"""Pure-logic tests for the localize page helpers."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from collab_splats.dashboard.config import LocalizationConfig
from collab_splats.dashboard.localize import (
    LocalizePage,
    SceneCache,
    camera_centers,
    preselect_method,
    subsample_step,
)
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.preproc import frames as fr

# Flat curated scene ids: reconstruction scene + the scene supplying the query video.
SCENE = "2026_05_07-birds-clip_03"
QUERY = "2026_05_08-birds-handheld"


def _page(tmp_path, source=None):
    """Build a LocalizePage with mocked source/worker (listing threads return empty)."""
    if source is None:
        source = MagicMock()
        source.list_scenes.return_value = []
        source.list_processed_scenes.return_value = []
    worker = MagicMock()
    worker.busy = False  # MagicMock attrs are truthy; the preview busy-gate needs a real flag
    return LocalizePage(base_dir=tmp_path, source=source, gpu_worker=worker, op_log=OperationLog())


def _join_threads(name):
    """Join every live background thread the page spawned under `name` (deterministic options)."""
    for t in threading.enumerate():
        if t.name == name:
            t.join(timeout=5)


def test_camera_centers_inverts_world_to_camera():
    # Camera at world (1, 2, 3), identity rotation: extrinsic t = -R @ C = -C
    ext = np.eye(4, dtype=np.float32)[None]
    ext[0, :3, 3] = [-1.0, -2.0, -3.0]
    centers = camera_centers(ext)
    np.testing.assert_allclose(centers[0], [1.0, 2.0, 3.0], atol=1e-6)


def test_subsample_step_thresholds():
    assert subsample_step(30) == 1
    assert subsample_step(60) == 1
    assert subsample_step(61) == 3
    assert subsample_step(300) == 3


def test_preselect_existing_db_wins():
    options, value = preselect_method(["disk"], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "disk"
    assert options == ["disk", "xfeat", "loma", "loma-g"]


def test_preselect_defaults_to_loma_when_no_db():
    _, value = preselect_method([], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "loma"


def test_preselect_prefers_loma_among_multiple_dbs():
    _, value = preselect_method(["disk", "loma"], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "loma"


def test_scene_cache_roundtrip():
    cache = SceneCache()
    assert cache.get(SCENE, "mesh") is None
    cache.put(SCENE, "mesh", object())
    assert cache.get(SCENE, "mesh") is not None
    cache.clear()
    assert cache.get(SCENE, "mesh") is None


def test_scene_cache_drop_kind_prefix():
    cache = SceneCache()
    cache.put(SCENE, "mesh", object())
    cache.put(SCENE, "localizer:loma-g", object())
    cache.put(QUERY, "localizer:disk", object())
    cache.drop_kind("localizer")
    assert cache.get(SCENE, "localizer:loma-g") is None
    assert cache.get(QUERY, "localizer:disk") is None
    assert cache.get(SCENE, "mesh") is not None  # CPU loads survive


def test_load_video_button_previews_current_frame(tmp_path, monkeypatch):
    """Explicit Load-video click fetches + shows the slider's frame immediately."""
    from pathlib import Path as _P

    import numpy as np

    page = _page(tmp_path)
    shown = []
    monkeypatch.setattr(page, "_ensure_local_query_video", lambda *a: _P("/dev/null"))
    monkeypatch.setattr(
        "collab_splats.preproc.extract_frame",
        lambda video, idx: np.full((4, 4, 3), idx, dtype=np.uint8),
    )
    monkeypatch.setattr(page, "_show_frame", lambda f: shown.append(int(f[0, 0, 0])))
    page.query_scene.options = [QUERY]
    page.query_scene.value = QUERY
    shown.clear()  # drop the on-select frame-0 preview
    page.frame_slider.end = 100
    page.frame_slider.value = 0  # no watcher fire (already 0) -> no debounce in flight
    token_before = page._preview_token
    page._on_load_video(None)
    # The handler spawns a thread; wait for the preview to land.
    for _ in range(50):
        if shown:
            break
        import time as _t

        _t.sleep(0.1)
    assert shown == [0]
    assert page._preview_token == token_before + 1


def test_load_video_button_requires_selection(tmp_path):
    page = _page(tmp_path)
    page._on_load_video(None)
    assert any("select a query scene" in line for line in page._op_log.log_lines)


def test_select_options_blank_first():
    from collab_splats.dashboard.localize import select_options

    opts = select_options(["a", "b"], "— pick —")
    assert next(iter(opts.values())) == ""  # blank entry first -> nothing auto-selected
    assert opts["a"] == "a" and opts["b"] == "b"


def test_scene_dropdowns_do_not_auto_cascade(tmp_path):
    """Populating a blank-first dropdown leaves the blank selected: no listing cascade."""
    from collab_splats.dashboard.localize import select_options

    page = _page(tmp_path)
    page.scene.options = select_options([SCENE], "— select scene —")
    # Blank/None both mean "nothing picked"; the watchers guard on falsy values.
    assert not page.scene.value


def test_scene_cache_evicts_oldest_mesh_beyond_keep():
    cache = SceneCache()
    for i in range(5):  # _KIND_KEEP["mesh"] == 3
        cache.put(f"2026_05_07-birds-v{i}", "mesh", f"m{i}")
    assert cache.get("2026_05_07-birds-v0", "mesh") is None
    assert cache.get("2026_05_07-birds-v1", "mesh") is None
    assert cache.get("2026_05_07-birds-v4", "mesh") == "m4"


def test_scene_cache_unbounded_kinds_untouched():
    cache = SceneCache()
    for i in range(5):
        cache.put(f"2026_05_07-birds-v{i}", "localizer:disk", i)
    assert cache.get("2026_05_07-birds-v0", "localizer:disk") == 0


def test_localize_set_busy_disables_widgets(tmp_path):
    page = _page(tmp_path)
    page.set_busy(True)
    for w in (
        page.run_btn,
        page.load_video_btn,
        page.scene,
        page.query_scene,
        page.method,
        page.append_db,
    ):
        assert w.disabled
    page.set_busy(False)
    assert not page.run_btn.disabled


def test_localize_sync_busy_refreshes_label_while_busy(tmp_path):
    page = _page(tmp_path)
    page._gpu.busy = True
    page._sync_busy()  # transition into busy
    page._op_log.current_op = "matching"
    page._sync_busy()  # already busy -> label-refresh branch
    assert "matching" in page.busy_note.object


def test_frame_slider_preview_latest_wins(tmp_path, monkeypatch):
    page = _page(tmp_path)
    shown = []
    monkeypatch.setattr(page, "_ensure_local_query_video", lambda *a: Path("/dev/null"))
    monkeypatch.setattr(
        "collab_splats.preproc.extract_frame",
        lambda video, idx: np.full((4, 4, 3), idx, dtype=np.uint8),
    )
    monkeypatch.setattr(page, "_show_frame", lambda f: shown.append(int(f[0, 0, 0])))
    page.query_scene.options = [QUERY]
    page.query_scene.value = QUERY
    shown.clear()  # ignore any on-select frame-0 preview
    page._preview_token = 2
    page._preview_frame(token=1, frame_idx=5, doc=None)  # superseded -> dropped
    page._preview_frame(token=2, frame_idx=9, doc=None)  # current -> shown
    assert shown == [9]
    assert any("frame 9 loaded" in line for line in page._op_log.log_lines)


def test_frame_slider_skipped_while_busy(tmp_path):
    page = _page(tmp_path)
    page._gpu.busy = True
    before = page._preview_token
    page._on_frame_slider(SimpleNamespace(new=7))
    assert page._preview_token == before  # no timer scheduled while a run is in flight


def test_build_panes_returns_fresh_objects_each_call(tmp_path):
    """Panes must be per-document: two builds share no pane objects (stale-doc bug class)."""
    page = _page(tmp_path)
    a = page._build_panes()
    b = page._build_panes()
    assert set(a) == {"matches_col", "vtk", "dist", "stats"}
    assert all(a[k] is not b[k] for k in a)


def test_show_frame_before_panes_is_pending_then_renders(tmp_path):
    """_show_frame before main() must not crash; the frame renders when panes build."""
    import panel as pn

    page = _page(tmp_path)
    assert page._panes is None
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    page._show_frame(frame)  # no panes yet -> state only
    assert page._state["left"] == "frame"
    page._panes = page._build_panes()
    page._render_state()
    assert isinstance(page._panes["matches_col"][0], pn.pane.Image)


def test_show_frame_downscales_to_thumbnail(tmp_path):
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


def test_render_browse_paints_header_and_scene(tmp_path, monkeypatch):
    """Browse render: header count in left column, all stored poses passed to the scene."""
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


def test_refresh_listings_wires_processed_to_scene_and_curated_to_query(tmp_path):
    """Scene = reconstructions (processed bucket); Query = any curated video. The two buckets
    are given DISJOINT contents here so a swap of the two listings cannot pass."""
    source = MagicMock()
    source.list_processed_scenes.return_value = [SCENE]
    source.list_scenes.return_value = [QUERY]
    page = _page(tmp_path, source=source)
    _join_threads("localize-list")
    assert set(page.scene.options.values()) == {"", SCENE}
    assert set(page.query_scene.options.values()) == {"", QUERY}


def test_refresh_listings_sets_options_on_the_ioloop(tmp_path, monkeypatch):
    """Options must be assigned inside a next-tick callback: mutating widgets from the listing
    thread writes to the Bokeh document off its IOLoop and silently drops updates."""
    import panel as pn

    class _FakeDoc:
        session_context = None  # panel's state.session_args probes this on a live curdoc

        def __init__(self):
            self.callbacks = []

        def add_next_tick_callback(self, cb):
            self.callbacks.append(cb)

    source = MagicMock()
    source.list_processed_scenes.return_value = [SCENE]
    source.list_scenes.return_value = [QUERY]
    doc = _FakeDoc()
    monkeypatch.setattr(pn.state, "curdoc", doc)
    page = _page(tmp_path, source=source)
    _join_threads("localize-list")
    assert not page.scene.options  # nothing set yet — the setter is queued, not run
    assert len(doc.callbacks) == 1
    doc.callbacks[0]()
    assert set(page.scene.options.values()) == {"", SCENE}
    assert set(page.query_scene.options.values()) == {"", QUERY}


def test_localize_scene_watcher_ignores_the_blank_option(tmp_path):
    """The blank '— select scene —' entry must not cascade a remote feature-DB listing."""
    page = _page(tmp_path)
    page._source.list_localization_dbs.reset_mock()
    page._on_scene(SimpleNamespace(new=""))
    _join_threads("db-list")
    page._source.list_localization_dbs.assert_not_called()


def test_run_provenance_names_the_query_video_file(tmp_path, monkeypatch):
    """Provenance is permanent (run_config.yaml -> zarr attrs): video_ref must identify the
    FILE, since a curated dir may hold more than one video, plus the scene id and frame."""
    import collab_splats.dashboard.pipeline as pipeline

    page = _page(tmp_path)
    page._source.list_localization_dbs.return_value = ["loma"]
    monkeypatch.setattr(page, "_load_browse", lambda *a, **k: None)
    monkeypatch.setattr(page, "_ensure_local_query_video", lambda s: tmp_path / s / "handheld.MP4")
    monkeypatch.setattr(page, "_build_result_figures", lambda *a, **k: {})
    monkeypatch.setattr(page, "_ensure_scene_mesh", lambda *a: None)
    seen = {}
    monkeypatch.setattr(pipeline, "run_localization", lambda **kw: seen.update(kw) or SimpleNamespace())
    submitted = []
    page._gpu.submit = lambda job, on_done, doc: submitted.append(job)

    page.scene.options = [SCENE]
    page.scene.value = SCENE
    page.query_scene.options = [QUERY]
    page.query_scene.value = QUERY
    page.frame_slider.end = 100
    page.frame_slider.value = 7
    page._on_run(None)

    assert len(submitted) == 1
    submitted[0]()  # the worker job builds provenance from the fetched video
    assert seen["provenance"] == {
        "video_ref": f"{QUERY}/handheld.MP4",
        "scene": QUERY,
        "frame_idx": 7,
    }


def test_scene_select_triggers_browse_load(tmp_path, monkeypatch):
    """Choosing a scene spawns the browse load with the preselected extractor."""
    import time

    page = _page(tmp_path)
    calls = []
    monkeypatch.setattr(page, "_load_browse", lambda scene, extractor, doc: calls.append((scene, extractor)))
    page._source.list_localization_dbs.return_value = ["loma"]
    page._on_scene(SimpleNamespace(new=SCENE))
    for _ in range(50):  # _on_scene runs its work() on a thread
        if calls:
            break
        time.sleep(0.1)
    assert calls == [(SCENE, "loma")]


def test_db_note_includes_localized_count(tmp_path):
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


def test_run_success_invalidates_browse_state(tmp_path, monkeypatch):
    page = _page(tmp_path)
    page._state["browse"] = (SimpleNamespace(extractor="loma", localized_extrinsics=np.zeros((1, 4, 4))), None)
    monkeypatch.setattr(page, "_render_state", lambda: None)
    figs = {"dist_fig": None, "match_figs": [], "stats_html": ""}
    page._handle_run_done((SimpleNamespace(), figs, None))
    assert page._state["browse"] is None
    assert page._state["left"] == "run"


def test_build_result_figures_is_pure(tmp_path, monkeypatch):
    """Figure building must be worker-safe: consumes the output, returns figs dict, touches no panes."""
    import matplotlib.figure

    page = _page(tmp_path)
    # Stub the plotting functions so no real matplotlib rendering happens
    monkeypatch.setattr(
        "collab_splats.localization.viz.plot_inlier_distribution",
        lambda ref_frame_indices, inlier_mask, n_frames=0, frame_sources=None: matplotlib.figure.Figure(),
    )
    monkeypatch.setattr(
        "collab_splats.localization.viz.plot_correspondences",
        lambda *a, **k: matplotlib.figure.Figure(),
    )
    # Minimal fake LocalizationRunOutput (mirrors test_run_localization's fixture fields)
    loc = SimpleNamespace(
        pose=np.eye(4, dtype=np.float32),
        n_correspondences=8,
        n_inliers=6,
        ref_frame_indices=np.zeros(0, dtype=np.int32),
        inlier_mask=np.zeros(0, dtype=bool),
        ranked_ref_frames=[],
    )
    out = SimpleNamespace(
        result=loc,
        query_frame=np.zeros((4, 4, 3), np.uint8),
        query_intrinsics=500.0 * np.eye(3, dtype=np.float32),
        intrinsics_source="estimated (experimental)",
        ref_image_paths=[],
        ref_extrinsics=np.eye(4, dtype=np.float32)[None],
        frame_sources=[],
    )
    figs = page._build_result_figures(out, LocalizationConfig(matcher="disk-lightglue"))
    assert set(figs) == {"dist_fig", "match_figs", "stats_html"}
    assert figs["match_figs"] == []  # no ref indices -> no correspondence figures


def test_build_result_figures_resolves_ref_arrays(tmp_path, monkeypatch):
    """Non-empty match path: each ranked ref resolves to an RGB array (images/ for reconstruction
    frames, disk for localized) and its correspondences arrive pre-sliced as plain arrays."""
    import cv2
    import matplotlib.figure

    page = _page(tmp_path)

    # Capture what plot_correspondences receives so we can assert the boundary wiring
    calls = []

    def _capture(query_image, ref_image, query_px, ref_px, inlier_mask=None, **kwargs):
        calls.append((ref_image, query_px, ref_px, inlier_mask))
        return matplotlib.figure.Figure()

    monkeypatch.setattr(
        "collab_splats.localization.viz.plot_inlier_distribution",
        lambda ref_frame_indices, inlier_mask, n_frames=0, frame_sources=None: matplotlib.figure.Figure(),
    )
    monkeypatch.setattr("collab_splats.localization.viz.plot_correspondences", _capture)

    # Real images/ store: the reconstruction ref frame resolves to a known RGB array
    store_pixels = np.full((4, 4, 3), 7, np.uint8)
    images_dir = tmp_path / "images"
    fr.write_frames(images_dir, [store_pixels], [{"frame_idx": 0}], {"video_path": "x"})

    # ref 0 = reconstruction (images/ branch); ref 1 = localized (disk branch — write a real JPG)
    localized_jpg = tmp_path / "localized_0001.jpg"
    cv2.imwrite(str(localized_jpg), np.zeros((4, 4, 3), np.uint8))

    # Real correspondence arrays: frame 0 owns two pairs, frame 1 owns one
    loc = SimpleNamespace(
        pose=np.eye(4, dtype=np.float32),
        n_correspondences=8,
        n_inliers=6,
        pts2d=np.array([[0, 0], [1, 1], [2, 2]], np.float32),
        pts2d_ref=np.array([[3, 3], [4, 4], [5, 5]], np.float32),
        ref_frame_indices=np.array([0, 0, 1], np.int32),
        inlier_mask=np.array([True, False, True]),
        ranked_ref_frames=[0, 1],
    )
    out = SimpleNamespace(
        result=loc,
        query_frame=np.zeros((4, 4, 3), np.uint8),
        query_intrinsics=500.0 * np.eye(3, dtype=np.float32),
        intrinsics_source="estimated (experimental)",
        ref_image_paths=["frame_000000.png", str(localized_jpg)],
        ref_extrinsics=np.eye(4, dtype=np.float32)[None],
        frame_sources=["reconstruction", "localized"],
    )
    figs = page._build_result_figures(out, LocalizationConfig(matcher="disk-lightglue"), images_dir=images_dir)

    # Both ranked refs produced a figure with per-frame pre-sliced correspondence arrays
    assert len(figs["match_figs"]) == 2
    assert [len(q_px) for _, q_px, _, _ in calls] == [2, 1]
    for ref_image, q_px, r_px, mask in calls:
        assert isinstance(ref_image, np.ndarray) and ref_image.ndim == 3
        assert len(q_px) == len(r_px) == len(mask)
    # Reconstruction frame came from images/ (known pixel value), not the localized JPG
    assert np.array_equal(calls[0][0], store_pixels)
