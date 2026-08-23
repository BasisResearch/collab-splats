"""overwrite=True on the localization DB must drop the stale zarr cache group.

Regression tests for the silent no-op: build_localization_db(overwrite=True) called
_build_localization_db with no overwrite notion, and from_feedforward always
cache-hits on an existing local_features/<extractor>/reconstruction group — so the
stale (payload-less) cache was reloaded untouched and verify()'s rebuild-once
branch crashed downstream.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import zarr

from collab_splats.wrapper import reconstructor as R
from collab_splats.wrapper.reconstructor import Reconstructor, _build_localization_db

REC_KEY = "local_features/loma/reconstruction"


def _make_stale_store(tmp_path: Path) -> Path:
    """Create a pointcloud.zarr holding a dummy stale reconstruction group."""
    ff = tmp_path / "pointcloud.zarr"
    store = zarr.open_group(str(ff), mode="a")
    grp = store.require_group(REC_KEY)
    grp.create_array("dummy", shape=(3,), dtype="float32")
    return ff


def _patch_heavy_deps(stack: ExitStack, ff: Path, seen: dict):
    """Stub the heavy inline deps of _build_localization_db; record cache state at call time."""

    # from_feedforward is where the cache check lives — capture whether the stale
    # reconstruction group still exists in the store at the moment it runs.
    def record_cache_state(*args, **kwargs):
        seen["rec_group_present"] = REC_KEY in zarr.open_group(str(ff), mode="r")
        return MagicMock()

    stack.enter_context(
        patch(
            "collab_splats.localization.localizer.CameraLocalizer.from_feedforward",
            side_effect=record_cache_state,
        )
    )
    stack.enter_context(
        patch(
            "collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr",
            return_value=MagicMock(),
        )
    )
    stack.enter_context(patch("collab_splats.localization.extractors.LocalMatcher", return_value=MagicMock()))
    frame_store = MagicMock()
    frame_store.frame_indices.return_value = [0, 1]
    stack.enter_context(patch("collab_splats.preproc.frame_store.FrameStore.open", return_value=frame_store))


def test_overwrite_true_drops_stale_group_before_rebuild(tmp_path):
    ff = _make_stale_store(tmp_path)
    frames = tmp_path / "frames.zarr"
    seen = {}
    with ExitStack() as stack:
        _patch_heavy_deps(stack, ff, seen)
        out = _build_localization_db(ff, "loma", frames, top_k=8, overwrite=True)
    # The stale group must be gone when from_feedforward runs, so its cache check misses
    assert seen["rec_group_present"] is False
    assert out == ff


def test_overwrite_false_keeps_existing_group(tmp_path):
    ff = _make_stale_store(tmp_path)
    frames = tmp_path / "frames.zarr"
    seen = {}
    with ExitStack() as stack:
        _patch_heavy_deps(stack, ff, seen)
        _build_localization_db(ff, "loma", frames, top_k=8, overwrite=False)
    # Default path is untouched: the existing group is still there for the cache hit
    assert seen["rec_group_present"] is True


def test_reconstructor_passes_overwrite_through(tmp_path):
    # Minimal config; base.yaml fills the rest (matcher=loma, top_k=8 defaults)
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "localization": {"enabled": True, "matcher": "loma"},
    }
    rec = Reconstructor(config)
    pc_zarr = rec.backend_dir / "pointcloud.zarr"
    pc_zarr.mkdir(parents=True)
    with patch.object(R, "_build_localization_db") as build:
        rec.build_localization_db(overwrite=True)
    build.assert_called_once_with(pc_zarr, "loma", rec.frames_zarr, top_k=8, overwrite=True)
