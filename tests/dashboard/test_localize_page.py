"""Pure-logic tests for the localize page helpers."""

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


def _page(tmp_path):
    """Build a LocalizePage with mocked source/worker (listing threads return empty)."""
    source = MagicMock()
    source.list_sessions.return_value = []
    source.list_field_sessions.return_value = []
    worker = MagicMock()
    worker.busy = False  # MagicMock attrs are truthy; the preview busy-gate needs a real flag
    return LocalizePage(base_dir=tmp_path, source=source, gpu_worker=worker, op_log=OperationLog())


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
    assert cache.get(("s", "v"), "mesh") is None
    cache.put(("s", "v"), "mesh", object())
    assert cache.get(("s", "v"), "mesh") is not None
    cache.clear()
    assert cache.get(("s", "v"), "mesh") is None


def test_scene_cache_drop_kind_prefix():
    cache = SceneCache()
    cache.put(("s", "v"), "mesh", object())
    cache.put(("s", "v"), "localizer:loma-g", object())
    cache.put(("s", "w"), "localizer:disk", object())
    cache.drop_kind("localizer")
    assert cache.get(("s", "v"), "localizer:loma-g") is None
    assert cache.get(("s", "w"), "localizer:disk") is None
    assert cache.get(("s", "v"), "mesh") is not None  # CPU loads survive


def test_load_video_button_previews_current_frame(tmp_path, monkeypatch):
    """Explicit Load-video click fetches + shows the slider's frame immediately."""
    from pathlib import Path as _P

    import numpy as np

    page = _page(tmp_path)
    shown = []
    monkeypatch.setattr(page, "_ensure_local_query_video", lambda *a: _P("/dev/null"))
    monkeypatch.setattr(
        "collab_splats.preproc.extract_frame_fast",
        lambda video, idx: np.full((4, 4, 3), idx, dtype=np.uint8),
    )
    monkeypatch.setattr(page, "_show_frame", lambda f: shown.append(int(f[0, 0, 0])))
    page.field_session.options = ["fs"]
    page.field_session.value = "fs"
    page.camera.options = ["rgb_0"]
    page.camera.value = "rgb_0"
    page.query_video.options = ["v.mp4"]
    page.query_video.value = "v.mp4"
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
    assert any("select a field session" in line for line in page._op_log.log_lines)


def test_select_options_blank_first():
    from collab_splats.dashboard.localize import select_options

    opts = select_options(["a", "b"], "— pick —")
    assert next(iter(opts.values())) == ""  # blank entry first -> nothing auto-selected
    assert opts["a"] == "a" and opts["b"] == "b"


def test_scene_dropdowns_do_not_auto_cascade(tmp_path):
    """Populating a blank-first dropdown leaves the blank selected: no listing cascade."""
    from collab_splats.dashboard.localize import select_options

    page = _page(tmp_path)
    page.scene_session.options = select_options(["s1"], "— select scene session —")
    # Blank/None both mean "nothing picked"; the watchers guard on falsy values.
    assert not page.scene_session.value


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


def test_localize_set_busy_disables_widgets(tmp_path):
    page = _page(tmp_path)
    page.set_busy(True)
    for w in (
        page.run_btn,
        page.scene_session,
        page.scene_video,
        page.field_session,
        page.camera,
        page.query_video,
        page.method,
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
        "collab_splats.preproc.extract_frame_fast",
        lambda video, idx: np.full((4, 4, 3), idx, dtype=np.uint8),
    )
    monkeypatch.setattr(page, "_show_frame", lambda f: shown.append(int(f[0, 0, 0])))
    page.field_session.options = ["fs"]
    page.field_session.value = "fs"
    page.camera.options = ["rgb_0"]
    page.camera.value = "rgb_0"
    page.query_video.options = ["v.mp4"]
    page.query_video.value = "v.mp4"
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


def test_build_result_figures_is_pure(tmp_path, monkeypatch):
    """Figure building must be worker-safe: consumes the output, returns figs dict, touches no panes."""
    import matplotlib.figure

    page = _page(tmp_path)
    # Stub the plotting functions so no real matplotlib rendering happens
    monkeypatch.setattr(
        "collab_splats.localization.viz.plot_inlier_distribution",
        lambda loc, n_frames=0, frame_sources=None: matplotlib.figure.Figure(),
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
        ref_frame_indices=None,
        inlier_mask=None,
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
    figs = page._build_result_figures(out, LocalizationConfig(extractor="disk"))
    assert set(figs) == {"dist_fig", "match_figs", "stats_html"}
    assert figs["match_figs"] == []  # no ref indices -> no correspondence figures
