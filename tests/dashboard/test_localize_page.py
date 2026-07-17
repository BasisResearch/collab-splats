"""Pure-logic tests for the localize page helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

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


def test_preselect_defaults_to_loma_g_when_no_db():
    _, value = preselect_method([], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "loma-g"


def test_preselect_prefers_loma_g_among_multiple_dbs():
    _, value = preselect_method(["disk", "loma-g"], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "loma-g"


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


def test_show_frame_downscales_to_thumbnail(tmp_path):
    page = _page(tmp_path)
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    page._show_frame(big)
    assert page._frame_pane.object.width <= 640
