"""Field-session browsing and remote localization-DB discovery."""
import collab_splats.dashboard.sources as sources
from collab_splats.dashboard.sources import SessionSource


class _FakeClient:
    """Returns canned rclone listings keyed by (bucket, path)."""

    remote_name = "collab-data"

    def __init__(self, listings):
        self._listings = listings

    def list_directory(self, bucket, path):
        key = (bucket, path)
        if key not in self._listings:
            raise RuntimeError(f"no such path: {key}")
        return self._listings[key]

    def _cmd(self, *args):
        return ["true"]


def _dirs(*names):
    return [{"Name": n, "IsDir": True} for n in names]


def _files(*names):
    return [{"Name": n, "IsDir": False} for n in names]


def test_list_field_sessions_filters_pattern():
    src = SessionSource(client=_FakeClient({
        ("fieldwork_curated", ""): _dirs(
            "2024_02_06-session_0001", "2024_02_07-session_0012", "reconstruction", "misc"
        ),
    }))
    assert src.list_field_sessions() == ["2024_02_06-session_0001", "2024_02_07-session_0012"]


def test_list_rgb_cameras_only():
    src = SessionSource(client=_FakeClient({
        ("fieldwork_curated", "2024_02_06-session_0001"): _dirs("rgb_1", "rgb_2", "thermal_1"),
    }))
    assert src.list_rgb_cameras("2024_02_06-session_0001") == ["rgb_1", "rgb_2"]


def test_list_camera_videos_filters_extensions():
    src = SessionSource(client=_FakeClient({
        ("fieldwork_curated", "2024_02_06-session_0001/rgb_1"):
            _files("a.mp4", "b.MOV", "notes.txt"),
    }))
    assert src.list_camera_videos("2024_02_06-session_0001", "rgb_1") == ["a.mp4", "b.MOV"]


def test_list_localization_dbs_returns_extractor_names():
    src = SessionSource(client=_FakeClient({
        ("fieldwork_processed",
         "reconstruction/2024_02_06/vid/feedforward.zarr/local_features"):
            _dirs("loma-g", "disk"),
    }))
    assert src.list_localization_dbs("2024_02_06", "vid") == ["disk", "loma-g"]


def test_list_localization_dbs_missing_path_returns_empty():
    src = SessionSource(client=_FakeClient({}))
    assert src.list_localization_dbs("2024_02_06", "vid") == []


def test_pull_processed_passes_exclude_flags(tmp_path, monkeypatch):
    captured = {}

    class _CmdClient(_FakeClient):
        def _cmd(self, *args):
            captured["args"] = args
            return ["true"]

    monkeypatch.setattr(sources.subprocess, "run", lambda cmd, check: None)
    src = SessionSource(client=_CmdClient({}))
    src.pull_processed("2024_02_06", "vid", tmp_path, excludes=("frames.zarr/**",))
    assert "--exclude" in captured["args"]
    assert "frames.zarr/**" in captured["args"]
