"""Tests for scripts/preprocess_gdrive_videos.py."""

import csv
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not an importable package, so load the module by file path
_SCRIPT = Path(__file__).parents[2] / "scripts" / "preprocess_gdrive_videos.py"
_spec = importlib.util.spec_from_file_location("preprocess_gdrive_videos", _SCRIPT)
preproc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preproc)


########
# sanitize
########


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("splats", "splats"),
        ("Goprosplat", "Goprosplat"),
        ("Phone pics and splat videos", "Phone_pics_and_splat_videos"),
        ("gopro-splat", "gopro_splat"),
        ("PXL_20260629_225753909.TS", "PXL_20260629_225753909_TS"),
        ("a  --  b", "a_b"),
        ("_leading and trailing_", "leading_and_trailing"),
        ("keep_underscores", "keep_underscores"),
    ],
)
def test_sanitize(raw, expected):
    assert preproc.sanitize(raw) == expected


########
# flat_dir_name
########


def test_flat_dir_name_uses_underscore_date_and_hyphen_delimiter():
    video = Path("/x/2026-07-15/Goprosplat/GH010228.mp4")
    assert preproc.flat_dir_name("2026-07-15", "Goprosplat", video) == "2026_07_15-Goprosplat-GH010228"


def test_flat_dir_name_preserves_parent_case():
    video = Path("/x/GH010228.mp4")
    lower = preproc.flat_dir_name("2026-07-15", "Goprosplat", video)
    upper = preproc.flat_dir_name("2026-07-15", "GoproSplat", video)
    assert lower != upper


def test_flat_dir_name_keeps_video_stem_verbatim():
    # Dots in the video name survive; only the parent folder is sanitized
    video = Path("/x/PXL_20260630_002106958.TS.mp4")
    assert (
        preproc.flat_dir_name("2026-06-29", "Phone pics and splat videos", video)
        == "2026_06_29-Phone_pics_and_splat_videos-PXL_20260630_002106958.TS"
    )


def test_flat_dir_name_keeps_hyphens_in_video_stem():
    # The stem is last, so its hyphens stay parseable via split("-", 2)
    video = Path("/x/clip-take-2.mp4")
    name = preproc.flat_dir_name("2026-07-15", "splats", video)
    assert name == "2026_07_15-splats-clip-take-2"
    assert name.split("-", 2) == ["2026_07_15", "splats", "clip-take-2"]


########
# plan_pairs
########


def _touch(path):
    """Create an empty file, making parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def tree(tmp_path):
    """Miniature capture tree mirroring the real gdrive-src shapes."""
    root = tmp_path / "gdrive-src"
    # Pair with an extension AND case difference: .mp4 edit, .MP4 original
    _touch(root / "2026-07-15" / "Goprosplat" / "GH010228.mp4")
    _touch(root / "2026-07-15" / "Goprosplat" / "src" / "GH010228.MP4")
    # Edit with no original in src/: skipped
    _touch(root / "2026-07-15" / "Goprosplat" / "GH010229.mp4")
    # Original with no edit: skipped
    _touch(root / "2026-06-29" / "GoproSplat" / "src" / "GH010221.MP4")
    # Spaces in the parent name, and a .mp4 edit against a .MOV original
    _touch(root / "2026-06-29" / "Phone pics and splat videos" / "IMG_4085.mp4")
    _touch(root / "2026-06-29" / "Phone pics and splat videos" / "src" / "IMG_4085.MOV")
    # Videos directly in a date folder with no src/ anywhere: camera originals, skipped
    _touch(root / "2026-06-03" / "GH010218.MP4")
    # Noise that must never be picked up
    _touch(root / "2026-07-15" / "Goprosplat" / ".DS_Store")
    _touch(root / "notes.txt")
    _touch(root / "scratch" / "whatever.mp4")
    return root


def test_plan_pairs_returns_only_videos_with_a_src_counterpart(tree):
    names = [p.name for p in preproc.plan_pairs(tree)]
    assert names == [
        "2026_06_29-Phone_pics_and_splat_videos-IMG_4085",
        "2026_07_15-Goprosplat-GH010228",
    ]


def test_plan_pairs_matches_across_case(tree):
    pair = next(p for p in preproc.plan_pairs(tree) if p.name.endswith("GH010228"))
    assert pair.edit.name == "GH010228.mp4"
    assert pair.source.name == "GH010228.MP4"


def test_plan_pairs_matches_across_extension(tree):
    pair = next(p for p in preproc.plan_pairs(tree) if p.name.endswith("IMG_4085"))
    assert pair.edit.suffix == ".mp4"
    assert pair.source.suffix == ".MOV"


def test_plan_pairs_ignores_undated_top_level_dirs(tree):
    assert all("scratch" not in p.edit.parts for p in preproc.plan_pairs(tree))


def test_plan_pairs_walks_videos_directly_in_a_date_folder(tmp_path):
    # The old plan_copies never looked here, so 2026-06-03/GH010218-220.MP4 were invisible
    root = tmp_path / "gdrive-src"
    _touch(root / "2026-06-03" / "GH010218.mp4")
    _touch(root / "2026-06-03" / "src" / "GH010218.MP4")
    assert [p.name for p in preproc.plan_pairs(root)] == ["2026_06_03-2026_06_03-GH010218"]


def test_plan_pairs_never_treats_a_src_video_as_an_edit(tmp_path):
    # A video inside src/ is a camera original even if src/src/ somehow existed
    root = tmp_path / "gdrive-src"
    _touch(root / "2026-07-15" / "Goprosplat" / "src" / "GH010228.MP4")
    assert preproc.plan_pairs(root) == []


def test_plan_pairs_raises_on_name_collision(tmp_path):
    # "gopro splat" and "gopro-splat" both sanitize to gopro_splat
    root = tmp_path / "gdrive-src"
    for parent in ("gopro splat", "gopro-splat"):
        _touch(root / "2026-07-15" / parent / "GH010228.mp4")
        _touch(root / "2026-07-15" / parent / "src" / "GH010228.MP4")
    with pytest.raises(ValueError, match="collision"):
        preproc.plan_pairs(root)


########
# copy_video
########


def test_copy_video_writes_original_filename(tmp_path):
    src = tmp_path / "GH010228.mp4"
    src.write_bytes(b"video")
    dest_dir = tmp_path / "out" / "2026_07_15-Goprosplat-GH010228"

    assert preproc.copy_video(src, dest_dir) == 5
    assert (dest_dir / "GH010228.mp4").read_bytes() == b"video"
    # No .partial sidecar is left behind
    assert list(dest_dir.iterdir()) == [dest_dir / "GH010228.mp4"]


def test_copy_video_skips_existing_same_size(tmp_path):
    src = tmp_path / "GH010228.mp4"
    src.write_bytes(b"video")
    dest_dir = tmp_path / "out" / "scene"

    preproc.copy_video(src, dest_dir)
    assert preproc.copy_video(src, dest_dir) == 0
    assert preproc.copy_video(src, dest_dir, force=True) == 5


def test_copy_video_replaces_truncated_destination(tmp_path):
    src = tmp_path / "GH010228.mp4"
    src.write_bytes(b"video")
    dest_dir = tmp_path / "out" / "scene"
    dest_dir.mkdir(parents=True)
    (dest_dir / "GH010228.mp4").write_bytes(b"vi")

    assert preproc.copy_video(src, dest_dir) == 5
    assert (dest_dir / "GH010228.mp4").read_bytes() == b"video"


########
# Idempotency
########


def test_source_fingerprint_records_size_and_mtime(tmp_path):
    src = tmp_path / "GH010228.MP4"
    src.write_bytes(b"video")
    fp = preproc.source_fingerprint(src)
    assert fp["size_bytes"] == 5
    assert fp["mtime"] == pytest.approx(src.stat().st_mtime)


def test_metadata_round_trips(tmp_path):
    dest = tmp_path / "2026_07_15-Goprosplat-GH010228"
    dest.mkdir(parents=True)
    video = Path("GH010228.mp4")
    preproc.write_metadata(dest, video, {"unique_id": "x", "source": {"size_bytes": 5}})
    assert preproc.metadata_path(dest, video).name == "GH010228_metadata.json"
    assert preproc.read_metadata(dest, video)["source"]["size_bytes"] == 5


def test_read_metadata_returns_none_when_absent(tmp_path):
    assert preproc.read_metadata(tmp_path, Path("GH010228.mp4")) is None


def test_needs_copy_is_true_on_a_fresh_destination(tmp_path):
    src = tmp_path / "src" / "GH010228.MP4"
    edit = tmp_path / "GH010228.mp4"
    _touch(src)
    edit.write_bytes(b"video")
    pair = preproc.Pair(edit, src, "scene")
    assert preproc.needs_copy(pair, tmp_path / "out") is True


def test_needs_copy_is_false_when_the_recorded_source_still_matches(tmp_path):
    # The trap: injection grows the curated mp4 past its source size, so a destination-size
    # check would re-copy forever. The gate keys off the fingerprint in the sidecar instead.
    src = tmp_path / "src" / "GH010228.MP4"
    edit = tmp_path / "GH010228.mp4"
    _touch(src)
    edit.write_bytes(b"video")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "GH010228.mp4").write_bytes(b"video plus an injected gpmd track")
    pair = preproc.Pair(edit, src, "scene")
    preproc.write_metadata(dest, edit, {"source": preproc.source_fingerprint(edit)})

    assert preproc.needs_copy(pair, dest) is False
    assert preproc.needs_copy(pair, dest, force=True) is True


def test_needs_copy_is_true_when_the_source_changed(tmp_path):
    src = tmp_path / "src" / "GH010228.MP4"
    edit = tmp_path / "GH010228.mp4"
    _touch(src)
    edit.write_bytes(b"video")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "GH010228.mp4").write_bytes(b"video")
    pair = preproc.Pair(edit, src, "scene")
    preproc.write_metadata(dest, edit, {"source": preproc.source_fingerprint(edit)})

    edit.write_bytes(b"a re-exported, different edit")
    assert preproc.needs_copy(pair, dest) is True
