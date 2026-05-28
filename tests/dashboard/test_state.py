from pathlib import Path

from collab_splats.dashboard.state import AppState


def test_appstate_defaults():
    state = AppState()
    assert state.output_dir is None
    assert state.video_path is None
    assert state.frames_zarr_path is None
    assert state.selected_indices == []
    assert state.feedforward_result is None
    assert state.feature_maps_path is None
    assert state.lifted_features_path is None


def test_appstate_watch_fires_on_output_dir_change():
    state = AppState()
    received = []
    state.param.watch(lambda e: received.append(e.new), "output_dir")
    state.output_dir = Path("/tmp/test_out")
    assert received == [Path("/tmp/test_out")]


def test_appstate_frames_zarr_path_accepts_path():
    state = AppState()
    p = Path("/workspace/outputs/birds/frames.zarr")
    state.frames_zarr_path = p
    assert state.frames_zarr_path == p


def test_appstate_selected_indices_accepts_list():
    state = AppState()
    state.selected_indices = [0, 5, 10, 15]
    assert state.selected_indices == [0, 5, 10, 15]


def test_appstate_feature_maps_path_accepts_path():
    state = AppState()
    p = Path("/workspace/outputs/birds/vggt_omega/features.zarr")
    state.feature_maps_path = p
    assert state.feature_maps_path == p
