"""Tests for scripts/flatten_dataset.py naming and tree-walk rules."""

import importlib.util
from pathlib import Path

import pytest

# scripts/ is not an importable package, so load the module by file path
_SCRIPT = Path(__file__).parents[2] / "scripts" / "flatten_dataset.py"
_spec = importlib.util.spec_from_file_location("flatten_dataset", _SCRIPT)
flatten_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flatten_dataset)


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
    assert flatten_dataset.sanitize(raw) == expected


########
# flat_dir_name
########


def test_flat_dir_name_uses_underscore_date_and_hyphen_delimiter():
    video = Path("/x/2026-07-15/Goprosplat/GH010228.mp4")
    assert flatten_dataset.flat_dir_name("2026-07-15", "Goprosplat", video) == "2026_07_15-Goprosplat-GH010228"


def test_flat_dir_name_preserves_parent_case():
    video = Path("/x/GH010228.mp4")
    lower = flatten_dataset.flat_dir_name("2026-07-15", "Goprosplat", video)
    upper = flatten_dataset.flat_dir_name("2026-07-15", "GoproSplat", video)
    assert lower != upper


def test_flat_dir_name_keeps_video_stem_verbatim():
    # Dots in the video name survive; only the parent folder is sanitized
    video = Path("/x/PXL_20260630_002106958.TS.mp4")
    assert (
        flatten_dataset.flat_dir_name("2026-06-29", "Phone pics and splat videos", video)
        == "2026_06_29-Phone_pics_and_splat_videos-PXL_20260630_002106958.TS"
    )


def test_flat_dir_name_keeps_hyphens_in_video_stem():
    # The stem is last, so its hyphens stay parseable via split("-", 2)
    video = Path("/x/clip-take-2.mp4")
    name = flatten_dataset.flat_dir_name("2026-07-15", "splats", video)
    assert name == "2026_07_15-splats-clip-take-2"
    assert name.split("-", 2) == ["2026_07_15", "splats", "clip-take-2"]


########
# plan_copies
########


def _touch(path):
    """Create an empty file, making parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def tree(tmp_path):
    """Miniature capture tree mirroring the real gdrive-src shapes."""
    root = tmp_path / "gdrive-src"
    # src-only folder: must be skipped
    _touch(root / "2026-06-29" / "GoproSplat" / "src" / "GH010221.MP4")
    # spaces in the parent name, also src-only
    _touch(root / "2026-06-29" / "Phone pics and splat videos" / "src" / "IMG_4085.MOV")
    # root-level videos alongside a src/ folder: only the root-level ones are taken
    _touch(root / "2026-07-15" / "Goprosplat" / "GH010228.mp4")
    _touch(root / "2026-07-15" / "Goprosplat" / "GH010229.mp4")
    _touch(root / "2026-07-15" / "Goprosplat" / "src" / "GH010228.MP4")
    # noise that must not be picked up
    _touch(root / "2026-07-15" / "Goprosplat" / ".DS_Store")
    _touch(root / "notes.txt")
    _touch(root / "scratch" / "whatever.mp4")
    return root


def test_plan_copies_skips_src_only_folders(tree):
    names = [name for _, name in flatten_dataset.plan_copies(tree)]
    assert names == ["2026_07_15-Goprosplat-GH010228", "2026_07_15-Goprosplat-GH010229"]


def test_plan_copies_ignores_undated_top_level_dirs(tree):
    videos = [v for v, _ in flatten_dataset.plan_copies(tree)]
    assert all("scratch" not in v.parts for v in videos)


def test_plan_copies_raises_on_name_collision(tmp_path):
    root = tmp_path / "gdrive-src"
    # "gopro splat" and "gopro-splat" both sanitize to gopro_splat
    _touch(root / "2026-07-15" / "gopro splat" / "GH010228.mp4")
    _touch(root / "2026-07-15" / "gopro-splat" / "GH010228.mp4")
    with pytest.raises(ValueError, match="collision"):
        flatten_dataset.plan_copies(root)


########
# copy_video
########


def test_copy_video_writes_original_filename(tmp_path):
    src = tmp_path / "GH010228.mp4"
    src.write_bytes(b"video")
    dest_dir = tmp_path / "out" / "2026_07_15-Goprosplat-GH010228"

    assert flatten_dataset.copy_video(src, dest_dir) == 5
    assert (dest_dir / "GH010228.mp4").read_bytes() == b"video"
    # No .partial sidecar is left behind
    assert list(dest_dir.iterdir()) == [dest_dir / "GH010228.mp4"]


def test_copy_video_skips_existing_same_size(tmp_path):
    src = tmp_path / "GH010228.mp4"
    src.write_bytes(b"video")
    dest_dir = tmp_path / "out" / "scene"

    flatten_dataset.copy_video(src, dest_dir)
    assert flatten_dataset.copy_video(src, dest_dir) == 0
    assert flatten_dataset.copy_video(src, dest_dir, force=True) == 5


def test_copy_video_replaces_truncated_destination(tmp_path):
    src = tmp_path / "GH010228.mp4"
    src.write_bytes(b"video")
    dest_dir = tmp_path / "out" / "scene"
    dest_dir.mkdir(parents=True)
    (dest_dir / "GH010228.mp4").write_bytes(b"vi")

    assert flatten_dataset.copy_video(src, dest_dir) == 5
    assert (dest_dir / "GH010228.mp4").read_bytes() == b"video"
