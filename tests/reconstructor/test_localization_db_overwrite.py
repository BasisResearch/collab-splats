"""Building the localization DB must drop the stale zarr cache group.

Regression tests for the silent no-op: the localize stage's overwrite called
_build_localization_db with no overwrite notion, and from_pointcloud cache-hits on an
existing local_features/<extractor>/reconstruction group — so the
stale (payload-less) cache was reloaded untouched and verify()'s rebuild-once
branch crashed downstream.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import zarr

from collab_splats import reconstructor as R
from collab_splats.reconstructor import Reconstructor, _build_localization_db

REC_KEY = "local_features/loma/reconstruction"


def _make_stale_store(tmp_path: Path) -> Path:
    """Create a pointcloud.zarr holding a dummy stale reconstruction group."""
    pc_zarr = tmp_path / "pointcloud.zarr"
    store = zarr.open_group(str(pc_zarr), mode="a")
    grp = store.require_group(REC_KEY)
    grp.create_array("dummy", shape=(3,), dtype="float32")
    return pc_zarr


def _patch_heavy_deps(stack: ExitStack, pc_zarr: Path, seen: dict):
    """Stub the heavy deps of _build_localization_db; record cache state at call time."""

    # from_pointcloud is where the cache check lives — capture whether the stale
    # reconstruction group still exists in the store at the moment it runs.
    def record_cache_state(*args, **kwargs):
        seen["rec_group_present"] = REC_KEY in zarr.open_group(str(pc_zarr), mode="r")
        return MagicMock()

    stack.enter_context(
        patch(
            "collab_splats.localization.localizer.CameraLocalizer.from_pointcloud",
            side_effect=record_cache_state,
        )
    )
    stack.enter_context(
        patch(
            "collab_splats.pointcloud.base.PointcloudResult.load_zarr",
            return_value=MagicMock(),
        )
    )
    stack.enter_context(patch("collab_splats.reconstructor.LocalMatcher", return_value=MagicMock()))


def _images_dir(tmp_path: Path) -> Path:
    """
    A two-frame images/ directory for _build_localization_db to list.

    - Only the filenames are read here (frame_indices and ids come from them); the images
      genexpr is never consumed because from_pointcloud is stubbed, so empty files do.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    for i in (0, 1):
        (images_dir / f"frame_{i:06d}.png").touch()
    return images_dir


def test_build_drops_stale_group_before_rebuild(tmp_path):
    pc_zarr = _make_stale_store(tmp_path)
    images_dir = _images_dir(tmp_path)
    seen = {}
    with ExitStack() as stack:
        _patch_heavy_deps(stack, pc_zarr, seen)
        _build_localization_db(pc_zarr, "loma", images_dir)
    # The stale group must be gone when from_pointcloud runs, so its cache check misses
    assert seen["rec_group_present"] is False


def test_build_refuses_a_missing_images_store_and_keeps_the_db(tmp_path):
    pc_zarr = _make_stale_store(tmp_path)
    seen = {}

    with ExitStack() as stack, pytest.raises(FileNotFoundError, match="no images/ store"):
        _patch_heavy_deps(stack, pc_zarr, seen)
        _build_localization_db(pc_zarr, "loma", tmp_path / "images")

    # The check runs before the drop: the existing group survives, and no build started
    assert REC_KEY in zarr.open_group(str(pc_zarr), mode="r")
    assert seen == {}


def test_localize_stage_always_rebuilds(tmp_path):
    """run() only calls localize() to (re)build, so the stale group is always dropped."""
    # Minimal config; base.yaml fills the rest (matcher=loma default)
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "localization": {"enabled": True, "matcher": "loma"},
    }
    rec = Reconstructor(config)
    pc_zarr = rec.backend_dir / "pointcloud.zarr"
    pc_zarr.mkdir(parents=True)
    with patch.object(R, "_build_localization_db") as build:
        rec.localize()
    build.assert_called_once_with(pc_zarr, "loma", rec.images_dir)
