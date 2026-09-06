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
    pc_zarr = tmp_path / "pointcloud.zarr"
    store = zarr.open_group(str(pc_zarr), mode="a")
    grp = store.require_group(REC_KEY)
    grp.create_array("dummy", shape=(3,), dtype="float32")
    return pc_zarr


def _patch_heavy_deps(stack: ExitStack, pc_zarr: Path, seen: dict):
    """Stub the heavy inline deps of _build_localization_db; record cache state at call time."""

    # from_feedforward is where the cache check lives — capture whether the stale
    # reconstruction group still exists in the store at the moment it runs.
    def record_cache_state(*args, **kwargs):
        seen["rec_group_present"] = REC_KEY in zarr.open_group(str(pc_zarr), mode="r")
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


def _images_dir(tmp_path: Path) -> Path:
    """
    A two-frame images/ directory for _build_localization_db to list.

    - Only the filenames are read here (frame_indices and ids come from them); the images
      genexpr is never consumed because from_feedforward is stubbed, so empty files do.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    for i in (0, 1):
        (images_dir / f"frame_{i:06d}.png").touch()
    return images_dir


def test_overwrite_true_drops_stale_group_before_rebuild(tmp_path):
    pc_zarr = _make_stale_store(tmp_path)
    images_dir = _images_dir(tmp_path)
    seen = {}
    with ExitStack() as stack:
        _patch_heavy_deps(stack, pc_zarr, seen)
        out = _build_localization_db(pc_zarr, "loma", images_dir, top_k=8, overwrite=True)
    # The stale group must be gone when from_feedforward runs, so its cache check misses
    assert seen["rec_group_present"] is False
    assert out == pc_zarr


def test_overwrite_false_keeps_existing_group(tmp_path):
    pc_zarr = _make_stale_store(tmp_path)
    images_dir = _images_dir(tmp_path)
    seen = {}
    with ExitStack() as stack:
        _patch_heavy_deps(stack, pc_zarr, seen)
        _build_localization_db(pc_zarr, "loma", images_dir, top_k=8, overwrite=False)
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
    build.assert_called_once_with(pc_zarr, "loma", rec.images_dir, top_k=8, overwrite=True)
