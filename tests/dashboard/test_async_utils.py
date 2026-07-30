"""Tests for the off-IOLoop helper and the shared serialized video fetch."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

from collab_splats.dashboard.async_utils import ensure_local_video, run_off_loop

SCENE = "2026_05_07-birds-clip_03"


def test_run_off_loop_applies_result_inline_when_no_doc():
    seen = []
    t = run_off_loop(lambda: 42, seen.append, label="t", doc=None)
    t.join(timeout=5)
    assert seen == [42]


def test_run_off_loop_swallows_fetch_error_and_skips_apply():
    seen = []

    def boom():
        raise RuntimeError("network down")

    t = run_off_loop(boom, seen.append, label="t", doc=None)
    t.join(timeout=5)
    assert seen == []  # apply not called, no exception propagated


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


########
# ensure_local_video: the ONE shared fetch both dashboard pages call
########


def _fetch_source(name="clip_03.mp4"):
    source = MagicMock()
    source.scene_video.return_value = name
    return source


def test_ensure_local_video_fast_path_skips_fetch(tmp_path):
    """An already-local video is returned without touching the network (multi-GB download)."""
    local = tmp_path / SCENE / "clip_03.mp4"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x")
    source = _fetch_source()
    got = ensure_local_video(SCENE, base_dir=tmp_path, source=source, op_log=MagicMock(), label="⬇ x")
    assert got == local
    source.fetch_video.assert_not_called()


def test_ensure_local_video_probes_the_name_from_the_scene_listing(tmp_path):
    """The probed filename comes from scene_video(), not a hardcoded guess."""
    local = tmp_path / SCENE / "handheld_take2.MP4"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x")
    source = _fetch_source("handheld_take2.MP4")
    got = ensure_local_video(SCENE, base_dir=tmp_path, source=source, op_log=MagicMock(), label="⬇ x")
    source.scene_video.assert_called_once_with(SCENE)
    assert got == local
    source.fetch_video.assert_not_called()


def test_ensure_local_video_caches_under_the_scene_dir(tmp_path):
    """Cache location is base_dir/<scene>/ — the same dir pipeline outputs live in, so the
    localize page cannot re-download a video the splats page already has."""
    source = _fetch_source()
    dests = []

    def fetch(scene, dest_dir, on_line=None):
        dests.append(Path(dest_dir))
        return Path(dest_dir) / "clip_03.mp4"

    source.fetch_video.side_effect = fetch
    got = ensure_local_video(SCENE, base_dir=tmp_path, source=source, op_log=MagicMock(), label="⬇ x")
    assert dests == [tmp_path / SCENE]
    assert got == tmp_path / SCENE / "clip_03.mp4"


def test_ensure_local_video_serializes_concurrent_fetches(tmp_path):
    """Two pages racing for the same video download it ONCE: the lock spans probe + fetch,
    so the loser blocks and then hits the exists() fast path (no partial-file decode)."""
    source = _fetch_source()
    calls = []
    first_inside = threading.Event()
    second_inside = threading.Event()
    release = threading.Event()

    def fetch(scene, dest_dir, on_line=None):
        calls.append(scene)
        (first_inside if len(calls) == 1 else second_inside).set()
        release.wait(5)
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "clip_03.mp4").write_bytes(b"x")
        return dest / "clip_03.mp4"

    source.fetch_video.side_effect = fetch

    def call():
        ensure_local_video(SCENE, base_dir=tmp_path, source=source, op_log=MagicMock(), label="⬇ x")

    t1 = threading.Thread(target=call, daemon=True)
    t1.start()
    assert first_inside.wait(5)  # first fetch is in flight, holding the lock
    t2 = threading.Thread(target=call, daemon=True)
    t2.start()
    # Unlocked, the second caller probes a not-yet-written file and starts its own download.
    assert not second_inside.wait(0.5)
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert calls == [SCENE]  # exactly one download
