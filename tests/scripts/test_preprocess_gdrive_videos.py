"""Tests for scripts/preprocess_gdrive_videos.py."""

import csv
import importlib.util
import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not an importable package, so load the module by file path
_SCRIPT = Path(__file__).parents[2] / "scripts" / "preprocess_gdrive_videos.py"
_spec = importlib.util.spec_from_file_location("preprocess_gdrive_videos", _SCRIPT)
preproc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preproc)

# solve_offset's default audio rate, Hz
RATE = 8000


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


_ROOT = Path("/gdrive-src")


def test_flat_dir_name_uses_underscore_date_and_hyphen_delimiter():
    video = _ROOT / "2026-07-15" / "Goprosplat" / "GH010228.mp4"
    assert preproc.flat_dir_name(video, _ROOT) == "2026_07_15-Goprosplat-GH010228"


def test_flat_dir_name_preserves_parent_case():
    lower = preproc.flat_dir_name(_ROOT / "2026-07-15" / "Goprosplat" / "GH010228.mp4", _ROOT)
    upper = preproc.flat_dir_name(_ROOT / "2026-07-15" / "GoproSplat" / "GH010228.mp4", _ROOT)
    assert lower != upper


def test_flat_dir_name_keeps_video_stem_verbatim():
    # Dots in the video name survive; only the directory components are sanitized
    video = _ROOT / "2026-06-29" / "Phone pics and splat videos" / "PXL_20260630_002106958.TS.mp4"
    assert preproc.flat_dir_name(video, _ROOT) == "2026_06_29-Phone_pics_and_splat_videos-PXL_20260630_002106958.TS"


def test_flat_dir_name_keeps_hyphens_in_video_stem():
    # The stem is last, so its hyphens stay parseable via split("-", 2)
    name = preproc.flat_dir_name(_ROOT / "2026-07-15" / "splats" / "clip-take-2.mp4", _ROOT)
    assert name == "2026_07_15-splats-clip-take-2"
    assert name.split("-", 2) == ["2026_07_15", "splats", "clip-take-2"]


def test_flat_dir_name_joins_every_path_component():
    # The audiomoth shape: four directory levels below the source root
    video = (
        _ROOT
        / "audiomoth-only-deployments"
        / "20260817-20260824"
        / "boston-charlesgateeast-riverbank"
        / "splat_videos"
        / "GH010259.mp4"
    )
    assert preproc.flat_dir_name(video, _ROOT) == (
        "audiomoth_only_deployments-20260817_20260824-" "boston_charlesgateeast_riverbank-splat_videos-GH010259"
    )


def test_flat_dir_name_for_a_video_directly_in_the_source_root():
    assert preproc.flat_dir_name(_ROOT / "GH010218.mp4", _ROOT) == "GH010218"


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("2026-07-22/splats/GH010234.mp4", "2026_07_22-splats-GH010234"),
        ("2024-07-13/GPM_SPLAT/GH010198.mp4", "2024_07_13-GPM_SPLAT-GH010198"),
        (
            "2026-06-29/Phone pics and splat videos/PXL_20260629_225753909.TS.mp4",
            "2026_06_29-Phone_pics_and_splat_videos-PXL_20260629_225753909.TS",
        ),
        (
            "2026-04-03/videos_for_splats/IMG_0005.mp4",
            "2026_04_03-videos_for_splats-IMG_0005",
        ),
        (
            "2025-07-17/GOPROC_Splat_plus/GH010209.mp4",
            "2025_07_17-GOPROC_Splat_plus-GH010209",
        ),
    ],
)
def test_flat_dir_name_reproduces_names_already_in_the_bucket(relative, expected):
    """Pin real paths to bucket folder names; push uses rclone copy, so a rename duplicates data."""
    assert preproc.flat_dir_name(_ROOT / relative, _ROOT) == expected


########
# plan_videos
########


def _touch(path):
    """Create an empty file, making parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_videos_skips_dotfiles(tmp_path):
    # AppleDouble "._X" sidecars and .DS_Store pass a suffix filter, so dotfiles are skipped
    _touch(tmp_path / "C0104.MP4")
    _touch(tmp_path / "._C0104.MP4")
    _touch(tmp_path / ".DS_Store")
    assert [v.name for v in preproc.find_videos(tmp_path)] == ["C0104.MP4"]


@pytest.fixture
def tree(tmp_path):
    """Miniature capture tree mirroring the real gdrive-src shapes."""
    root = tmp_path / "gdrive-src"
    # Pair with an extension AND case difference: .mp4 edit, .MP4 original
    _touch(root / "2026-07-15" / "Goprosplat" / "GH010228.mp4")
    _touch(root / "2026-07-15" / "Goprosplat" / "src" / "GH010228.MP4")
    # No src/ counterpart: an un-edited camera original, curated on its own
    _touch(root / "2026-07-15" / "Goprosplat" / "GH010229.mp4")
    # A video left in src/ with nothing above it: an orphan, skipped and logged
    _touch(root / "2026-06-29" / "GoproSplat" / "src" / "GH010221.MP4")
    # Spaces in the parent name, and a .mp4 edit against a .MOV original
    _touch(root / "2026-06-29" / "Phone pics and splat videos" / "IMG_4085.mp4")
    _touch(root / "2026-06-29" / "Phone pics and splat videos" / "src" / "IMG_4085.MOV")
    # A video directly in a date folder with no src/ anywhere: another original
    _touch(root / "2026-06-03" / "GH010218.MP4")
    # An undated top-level folder: walked like any other, since only the src/ rule is assumed
    _touch(root / "scratch" / "whatever.mp4")
    _touch(root / "scratch" / "src" / "whatever.MP4")
    # Noise that must never be picked up
    _touch(root / "2026-07-15" / "Goprosplat" / ".DS_Store")
    _touch(root / "notes.txt")
    return root


def test_plan_videos_returns_every_video_outside_src(tree):
    # Both edits and un-edited originals are in scope; only what sits inside src/ is not
    assert [name for _, _, name in preproc.plan_videos(tree)] == [
        "2026_06_03-GH010218",
        "2026_06_29-Phone_pics_and_splat_videos-IMG_4085",
        "2026_07_15-Goprosplat-GH010228",
        "2026_07_15-Goprosplat-GH010229",
        "scratch-whatever",
    ]


def test_plan_videos_pairs_an_edit_with_its_source(tree):
    video, source, _ = next(e for e in preproc.plan_videos(tree) if e[2].endswith("GH010228"))
    assert video.name == "GH010228.mp4"
    assert source.name == "GH010228.MP4"


def test_plan_videos_gives_an_original_a_null_source(tree):
    # No src/ counterpart means the video is its own source: nothing to align, nothing to inject
    video, source, _ = next(e for e in preproc.plan_videos(tree) if e[2].endswith("GH010229"))
    assert video.name == "GH010229.mp4"
    assert source is None


def test_plan_videos_matches_across_case(tree):
    video, source, _ = next(e for e in preproc.plan_videos(tree) if e[2].endswith("GH010228"))
    assert video.suffix == ".mp4"
    assert source.suffix == ".MP4"


def test_plan_videos_matches_across_extension(tree):
    video, source, _ = next(e for e in preproc.plan_videos(tree) if e[2].endswith("IMG_4085"))
    assert video.suffix == ".mp4"
    assert source.suffix == ".MOV"


def test_plan_videos_walks_undated_top_level_dirs(tree):
    # No date gate: a folder is walked whatever it is called
    assert "scratch-whatever" in [name for _, _, name in preproc.plan_videos(tree)]


def test_plan_videos_walks_videos_directly_in_a_date_folder(tmp_path):
    # Videos sitting directly in a date folder are curated too
    root = tmp_path / "gdrive-src"
    _touch(root / "2026-06-03" / "GH010218.mp4")
    _touch(root / "2026-06-03" / "src" / "GH010218.MP4")
    assert [name for _, _, name in preproc.plan_videos(root)] == ["2026_06_03-GH010218"]


def test_plan_videos_walks_arbitrarily_deep_trees(tmp_path):
    # The audiomoth-only-deployments shape: four levels, and a .mov edit on a .MP4 original
    root = tmp_path / "gdrive-src"
    deep = root / "audiomoth-only-deployments" / "20260810-20260831" / "boston-ringerpark-west" / "splat_videos"
    _touch(deep / "GH010247.mov")
    _touch(deep / "src" / "GH010247.MP4")
    expected = "audiomoth_only_deployments-20260810_20260831-boston_ringerpark_west-splat_videos-GH010247"
    assert [name for _, _, name in preproc.plan_videos(root)] == [expected]


def test_plan_videos_skips_a_video_orphaned_inside_src(tmp_path):
    # An orphan inside src/ is never curated: its flat name would bake in "src"
    root = tmp_path / "gdrive-src"
    _touch(root / "2026-07-15" / "Goprosplat" / "src" / "GH010228.MP4")
    assert preproc.plan_videos(root) == []


def test_plan_videos_logs_every_orphan_by_name(tmp_path, caplog):
    # The whole point of skipping them: a human has to see which files need filing
    root = tmp_path / "gdrive-src"
    _touch(root / "2026-06-03" / "src" / "GH010220.MP4")
    _touch(root / "2026-07-22" / "splats" / "src" / "GH010233.MP4")
    with caplog.at_level("WARNING"):
        preproc.plan_videos(root)
    assert "GH010220.MP4" in caplog.text
    assert "GH010233.MP4" in caplog.text


def test_plan_videos_prunes_src_at_any_depth(tmp_path):
    # GH010252 sits in a src/ four levels down with no video above it: an orphan, not an edit
    root = tmp_path / "gdrive-src"
    deep = root / "audiomoth-only-deployments" / "20260817-20260824" / "site" / "splat_videos"
    _touch(deep / "src" / "GH010252.MP4")
    assert preproc.plan_videos(root) == []


def test_plan_videos_never_pairs_two_appledouble_files(tmp_path):
    # Matching AppleDouble ._X.MP4 files in both places must not pair as footage
    root = tmp_path / "gdrive-src"
    folder = root / "2024-07-09" / "SplatsSD"
    _touch(folder / "._C0104.MP4")
    _touch(folder / "src" / "._C0104.MP4")
    assert preproc.plan_videos(root) == []


def test_plan_videos_raises_on_name_collision(tmp_path):
    # "gopro splat" and "gopro-splat" both sanitize to gopro_splat
    root = tmp_path / "gdrive-src"
    for parent in ("gopro splat", "gopro-splat"):
        _touch(root / "2026-07-15" / parent / "GH010228.mp4")
        _touch(root / "2026-07-15" / parent / "src" / "GH010228.MP4")
    with pytest.raises(ValueError, match="collision"):
        preproc.plan_videos(root)


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
    assert preproc.needs_copy(edit, tmp_path / "out") is True


def test_needs_copy_is_false_when_the_recorded_source_still_matches(tmp_path):
    # Injection grows the curated mp4, so the gate keys off the sidecar fingerprint, not dest size
    src = tmp_path / "src" / "GH010228.MP4"
    edit = tmp_path / "GH010228.mp4"
    _touch(src)
    edit.write_bytes(b"video")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "GH010228.mp4").write_bytes(b"video plus an injected gpmd track")
    preproc.write_metadata(dest, edit, {"source": preproc.source_fingerprint(edit)})

    assert preproc.needs_copy(edit, dest) is False
    assert preproc.needs_copy(edit, dest, force=True) is True


def test_needs_copy_is_true_when_the_source_changed(tmp_path):
    src = tmp_path / "src" / "GH010228.MP4"
    edit = tmp_path / "GH010228.mp4"
    _touch(src)
    edit.write_bytes(b"video")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "GH010228.mp4").write_bytes(b"video")
    preproc.write_metadata(dest, edit, {"source": preproc.source_fingerprint(edit)})

    edit.write_bytes(b"a re-exported, different edit")
    assert preproc.needs_copy(edit, dest) is True


########
# DMS formatting
########


@pytest.mark.parametrize(
    "value,axis,expected",
    [
        (42.3532, "lat", "42 deg 21' 11.52\" N"),
        (-71.0659, "lon", "71 deg 3' 57.24\" W"),
        (-33.8688, "lat", "33 deg 52' 7.68\" S"),
        (151.2093, "lon", "151 deg 12' 33.48\" E"),
        (0.0, "lat", "0 deg 0' 0.00\" N"),
        (0.0, "lon", "0 deg 0' 0.00\" E"),
        (42.34999986, "lat", "42 deg 21' 0.00\" N"),
        (41.9999999, "lat", "42 deg 0' 0.00\" N"),
    ],
)
def test_format_dms(value, axis, expected):
    assert preproc.format_dms(value, axis) == expected


def test_format_dms_never_emits_a_leading_minus():
    # The hemisphere letter carries the sign; a minus as well would be double-signed
    assert "-" not in preproc.format_dms(-71.0659, "lon")


def test_format_gps_joins_both_axes():
    assert preproc.format_gps(42.3532, -71.0659) == "42 deg 21' 11.52\" N, 71 deg 3' 57.24\" W"


@pytest.mark.parametrize("lat,lon", [(None, -71.0659), (42.3532, None), (None, None)])
def test_format_gps_is_blank_without_a_fix(lat, lon):
    # Blank, never 0,0 and never a sentinel string
    assert preproc.format_gps(lat, lon) == ""


########
# CSV index
########


def _curated(root, unique_id, stem, gps):
    """Write a minimal curated folder with just the JSON sidecar the index reads."""
    folder = root / unique_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}_metadata.json").write_text(json.dumps({"unique_id": unique_id, "gps": gps}))
    return folder


def test_index_rows_are_sorted_by_unique_id(tmp_path):
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    _curated(tmp_path, "2026_06_29-splats-GH010221", "GH010221", {"latitude": 42.0, "longitude": -71.0})
    rows = preproc.index_rows(tmp_path)
    assert [r[0] for r in rows] == ["2026_06_29-splats-GH010221", "2026_07_22-splats-GH010234"]


def test_index_rows_format_gps_as_dms(tmp_path):
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    assert preproc.index_rows(tmp_path)[0][1] == "42 deg 21' 11.52\" N, 71 deg 3' 57.24\" W"


def test_index_rows_leave_gps_blank_without_a_fix(tmp_path):
    _curated(tmp_path, "2026_06_29-Phone-PXL_20260629_225753909.TS", "PXL_20260629_225753909.TS", None)
    assert preproc.index_rows(tmp_path)[0][1] == ""


def test_write_index_has_exactly_two_columns(tmp_path):
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    path = preproc.write_index(tmp_path)
    assert path.name == "index.csv"
    rows = list(csv.reader(io.StringIO(path.read_text())))
    assert rows[0] == ["unique_id", "gps"]
    assert len(rows[1]) == 2


def test_write_index_quotes_the_inch_mark_so_it_round_trips(tmp_path):
    # The DMS string contains both a comma and a double quote; csv must survive both
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    path = preproc.write_index(tmp_path)
    rows = list(csv.reader(io.StringIO(path.read_text())))
    assert rows[1][1] == "42 deg 21' 11.52\" N, 71 deg 3' 57.24\" W"


def test_write_index_survives_a_partial_run(tmp_path):
    # Rebuilding from sidecars means an untouched clip still appears in the index
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    _curated(tmp_path, "2026_06_29-splats-GH010221", "GH010221", None)
    preproc.write_index(tmp_path)
    rows = list(csv.reader(io.StringIO((tmp_path / "index.csv").read_text())))
    assert len(rows) == 3  # header + 2 clips


########
# Alignment
########


def _rng():
    """Deterministic generator so alignment tests never flake."""
    return np.random.default_rng(20260811)


def test_solve_offset_recovers_a_known_trim():
    rng = _rng()
    body = rng.standard_normal(RATE * 3).astype(np.float32)
    head = rng.standard_normal(RATE).astype(np.float32)
    tail = rng.standard_normal(RATE).astype(np.float32)
    source = np.concatenate([head, body, tail])

    result = preproc.solve_offset(body, source)
    assert result["ok"] is True
    assert result["r"] == pytest.approx(1.0, abs=1e-4)
    # Within one audio sample of the true 1.0 s offset
    assert abs(result["offset_s"] - 1.0) <= 1.0 / RATE


def test_solve_offset_is_unaffected_by_a_gain_change():
    # Normalized correlation is scale-free, so a level difference must not lower r
    rng = _rng()
    body = rng.standard_normal(RATE * 2).astype(np.float32)
    source = np.concatenate([rng.standard_normal(RATE).astype(np.float32), body])

    result = preproc.solve_offset(body * 0.25, source)
    assert result["ok"] is True
    assert result["r"] == pytest.approx(1.0, abs=1e-4)


def test_solve_offset_rejects_uncorrelated_audio():
    rng = _rng()
    edit = rng.standard_normal(RATE).astype(np.float32)
    source = rng.standard_normal(RATE * 5).astype(np.float32)

    result = preproc.solve_offset(edit, source)
    assert result["ok"] is False
    assert result["r"] < 0.95


def test_solve_offset_rejects_an_edit_longer_than_its_source():
    rng = _rng()
    edit = rng.standard_normal(RATE * 5).astype(np.float32)
    source = rng.standard_normal(RATE).astype(np.float32)

    result = preproc.solve_offset(edit, source)
    assert result["ok"] is False
    assert result["offset_s"] == 0.0


def test_solve_offset_rejects_silence():
    # A constant signal has zero variance, so correlation is undefined rather than perfect
    source = np.zeros(RATE * 3, dtype=np.float32)
    result = preproc.solve_offset(np.zeros(RATE, dtype=np.float32), source)
    assert result["ok"] is False


def test_solve_offset_survives_a_silent_source_window():
    # A silent source window must score 0, not nan, so it cannot win the argmax
    rng = _rng()
    body = rng.standard_normal(RATE).astype(np.float32)
    source = np.concatenate([np.zeros(RATE, dtype=np.float32), body])

    result = preproc.solve_offset(body, source)
    assert result["ok"] is True
    assert abs(result["offset_s"] - 1.0) <= 1.0 / RATE
    assert np.isfinite(result["r"])


def test_solve_offset_threshold_is_overridable():
    rng = _rng()
    edit = rng.standard_normal(RATE).astype(np.float32)
    source = rng.standard_normal(RATE * 5).astype(np.float32)

    assert preproc.solve_offset(edit, source, min_r=0.0)["ok"] is True


def test_solve_offset_recovers_a_lag_that_overruns_the_source_end():
    # AAC re-encode pads the edit, so the true lag sits just past len(source) - len(edit)
    rng = _rng()
    body = rng.standard_normal(RATE * 3).astype(np.float32)
    head = rng.standard_normal(RATE).astype(np.float32)
    source = np.concatenate([head, body])
    # A short tail with no counterpart in the source, standing in for the encoder's padding
    edit = np.concatenate([body, rng.standard_normal(400).astype(np.float32)])
    # The true lag exceeds what a strictly-fitting search could reach
    assert RATE > len(source) - len(edit)

    result = preproc.solve_offset(edit, source)
    assert result["ok"] is True
    assert abs(result["offset_s"] - 1.0) <= 1.0 / RATE


def test_solve_offset_rejects_an_edit_longer_than_the_source_plus_tolerance():
    # The tolerance widens the search; it must not become a way to accept nonsense
    rng = _rng()
    edit = rng.standard_normal(RATE * 5).astype(np.float32)
    source = rng.standard_normal(RATE).astype(np.float32)

    result = preproc.solve_offset(edit, source)
    assert result["ok"] is False
    assert result["offset_s"] == 0.0


########
# Metadata extraction
########

# One exiftool object keyed Main:/DocN:/DocN-M:, shaped like GH010218.MP4 with shortened values
_DUMP = [
    {
        "SourceFile": "/x/src/GH010218.MP4",
        "Main:Model": "GoPro Max",
        "Main:FirmwareVersion": "H19.03.02.00.00",
        "Main:CreateDate": "2026:06:03 14:18:06",
        "Main:MediaCreateDate": "2026:06:03 14:18:06",
        "Main:FieldOfView": "L",
        "Main:LensProjection": "GPRO",
        "Main:ElectronicImageStabilization": "HS EIS",
        "Main:GPSCoordinates": "42.3528 -71.0655",
        "Main:GPSLatitude": 42.3528,
        "Main:GPSLongitude": -71.0655,
        # Chunk 1: no satellite lock yet, so every fix in the window reads 0,0
        "Doc1:SampleTime": 0.0,
        "Doc1:SampleDuration": 1.001,
        "Doc1:GPSDateTime": "2026:06:03 17:42:53.970",
        "Doc1:GPSLatitude": 0.0,
        "Doc1:GPSLongitude": 0.0,
        "Doc1:GPSAltitude": 15.445,
        "Doc1-1:GPSLatitude": 0.0,
        "Doc1-1:GPSLongitude": 0.0,
        "Doc1-1:GPSAltitude": 15.48,
        # Chunk 2: locked
        "Doc2:SampleTime": 1.001,
        "Doc2:SampleDuration": 1.001,
        "Doc2:GPSDateTime": "2026:06:03 17:42:54.970",
        "Doc2:GPSLatitude": 42.3532,
        "Doc2:GPSLongitude": -71.0659,
        "Doc2:GPSAltitude": 15.52,
        "Doc2-1:GPSLatitude": 42.3533,
        "Doc2-1:GPSLongitude": -71.0660,
        "Doc2-1:GPSAltitude": 15.55,
        "Doc3:SampleTime": 2.002,
        "Doc3:SampleDuration": 1.001,
        "Doc3:GPSLatitude": 42.3534,
        "Doc3:GPSLongitude": -71.0661,
        "Doc3:GPSAltitude": 15.60,
        "Doc4:SampleTime": 0.0,
        "Doc4:SampleDuration": 0.0,
        "Doc4:Model": "GoPro Max",
        "Doc4:SerialNumber": "C3441325104321",
        "Doc4:FirmwareVersion": "H19.03.02.00.00",
    }
]


def test_iter_documents_groups_by_key_prefix():
    main, chunks = preproc.iter_documents(_DUMP)
    assert main["Model"] == "GoPro Max"
    # A key with no prefix at all is a whole-file fact like any other Main tag
    assert main["SourceFile"] == "/x/src/GH010218.MP4"
    assert [number for number, _, _ in chunks] == [1, 2, 3, 4]
    assert chunks[0][1]["SampleTime"] == 0.0
    assert chunks[1][2] == [{"GPSLatitude": 42.3533, "GPSLongitude": -71.0660, "GPSAltitude": 15.55}]


def test_iter_documents_orders_chunks_numerically_not_lexically():
    # Doc10 sorts before Doc2 as text, which would put the samples on the wrong time axis
    dump = [
        {
            "Doc10:SampleTime": 9.009,
            "Doc2:SampleTime": 1.001,
            "Doc2-2:GPSLatitude": 42.3533,
            "Doc2-1:GPSLatitude": 42.3532,
        }
    ]
    _, chunks = preproc.iter_documents(dump)
    assert [number for number, _, _ in chunks] == [2, 10]
    # And the sub-samples run in their own numeric order within the parent chunk
    assert [sub["GPSLatitude"] for sub in chunks[0][2]] == [42.3532, 42.3533]


def test_exif_dump_asks_for_the_binary_payloads(monkeypatch):
    # Without -b, wide IMU tags come back as a binary placeholder string instead of numbers
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "[]", "")

    monkeypatch.setattr(preproc.subprocess, "run", fake_run)
    assert preproc.exif_dump(Path("/x/src/GH010218.MP4")) == []
    assert "-b" in commands[0]
    assert "-ee" in commands[0] and "-G3" in commands[0] and "-n" in commands[0]


def test_static_tags_reads_the_main_group():
    tags = preproc.static_tags(_DUMP)
    assert tags["Model"] == "GoPro Max"
    assert tags["FirmwareVersion"] == "H19.03.02.00.00"
    assert tags["CreateDate"] == "2026:06:03 14:18:06"
    assert tags["GPSCoordinates"] == "42.3528 -71.0655"


def test_static_tags_prefers_the_main_group_over_a_chunk_of_the_same_name():
    # A per-chunk GPSAltitude must not override the container's own value
    dump = [{"Main:GPSAltitude": 5.2, "Doc1:SampleTime": 0.0, "Doc1:GPSAltitude": 15.445}]
    assert preproc.static_tags(dump)["GPSAltitude"] == 5.2


def test_static_tags_omits_a_tag_only_the_chunks_carry():
    # Tags that live only on GPMF chunks must not be reported as whole-file values
    tags = preproc.static_tags(_DUMP)
    assert "GPSAltitude" not in tags
    assert "SerialNumber" not in tags


def test_first_fix_skips_the_unlocked_zero_zero_sample():
    # A GoPro emits 0,0 before satellite lock, which is not a real fix
    fix = preproc.first_fix(_DUMP)
    assert fix["latitude"] == pytest.approx(42.3532)
    assert fix["longitude"] == pytest.approx(-71.0659)
    assert fix["source_time"] == pytest.approx(1.001)


def test_first_fix_returns_the_first_chunk_not_the_last():
    # The first chunk's fix is the clip's location, not the last chunk's
    dump = [
        {
            "Doc1:SampleTime": 0.0,
            "Doc1:SampleDuration": 1.001,
            "Doc1:GPSLatitude": 42.3527878,
            "Doc1:GPSLongitude": -71.065473,
            "Doc2:SampleTime": 1.001,
            "Doc2:SampleDuration": 1.001,
            "Doc2:GPSLatitude": 42.3527,
            "Doc2:GPSLongitude": -71.0653,
            "Doc3:SampleTime": 2.002,
            "Doc3:SampleDuration": 1.001,
            "Doc3:GPSLatitude": 42.3526494,
            "Doc3:GPSLongitude": -71.0651,
        }
    ]
    fix = preproc.first_fix(dump)
    assert fix["latitude"] == pytest.approx(42.3527878)
    assert fix["source_time"] == 0.0


def test_first_fix_returns_none_when_nothing_locked():
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:GPSLatitude": 0.0, "Doc1:GPSLongitude": 0.0}]
    assert preproc.first_fix(dump) is None


def test_gps_fixes_requires_chunk_timing():
    with pytest.raises(KeyError, match="SampleDuration"):
        preproc.gps_fixes((1, {"SampleTime": 0.0, "GPSLatitude": 42.0, "GPSLongitude": -71.0}, []))


def test_first_fix_reads_a_container_location_when_there_is_no_track():
    # Pixel and iPhone carry a single point and no GPS track at all
    dump = [{"Main:GPSLatitude": 42.3532, "Main:GPSLongitude": -71.0659}]
    fix = preproc.first_fix(dump)
    assert fix["latitude"] == pytest.approx(42.3532)
    assert fix["source_time"] == 0.0


def test_first_fix_prefers_a_timed_chunk_fix_over_the_container_fix():
    # The untimed, rounded container coordinate is only a fallback for clips with no GPMF track
    assert preproc.first_fix(_DUMP)["latitude"] == pytest.approx(42.3532)


def test_first_fix_honours_the_trim_window():
    # A fix before the cut starts is not where this clip was shot
    assert preproc.first_fix(_DUMP, start_s=3.5) is None


def test_first_fix_reads_a_sub_sample_inside_the_chunk_window():
    # Untimed DocN-M fixes are spread across their chunk, so chunk 2's lands at 1.5015 s
    fix = preproc.first_fix(_DUMP, start_s=1.5)
    assert fix["latitude"] == pytest.approx(42.3533)
    assert fix["source_time"] == pytest.approx(1.5015)


def test_gps_payload_falls_back_to_the_whole_source_when_the_cut_has_no_fix():
    # No locked fix at or after 4.2 s in _DUMP, so this exercises the whole-source fallback
    payload = preproc.gps_payload(_DUMP, {"offset_s": 4.2, "r": 0.997, "ok": True})
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["gps_source_anchored"] is True


def test_gps_payload_records_a_missing_fix_as_null():
    payload = preproc.gps_payload([{"Main:Model": "Pixel 9 Pro"}], {"offset_s": 0.0, "r": 0.99, "ok": True})
    assert payload["gps"] is None
    # No fix anywhere, so the flag must not claim an approximate fix exists
    assert payload["gps_source_anchored"] is False


def test_gps_payload_marks_a_fix_inside_the_cut_as_exact():
    # A fix at 1.001 s lies inside a cut starting at 0.5 s, so it is not source-anchored
    payload = preproc.gps_payload(_DUMP, {"offset_s": 0.5, "r": 0.997, "ok": True})
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["gps_source_anchored"] is False


def test_gps_payload_does_not_anchor_a_container_only_fix():
    # A phone clip's untimed container fix is not a lost timed fix, so it is not source-anchored
    dump = [{"Main:Model": "Pixel 9 Pro", "Main:GPSLatitude": 42.3532, "Main:GPSLongitude": -71.0659}]
    payload = preproc.gps_payload(dump, {"offset_s": 4.2, "r": 0.99, "ok": True})
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["gps_source_anchored"] is False


def test_gps_payload_records_the_gpmd_chunk_residual():
    # The injected 1 Hz track starts on the first chunk at or after the cut; record the residual
    payload = preproc.gps_payload(_DUMP, {"offset_s": 0.4, "r": 0.997, "ok": True})
    # _DUMP has chunks at 0.0 and 1.001; the first at or after 0.4 is 1.001
    assert payload["gpmd_first_chunk_s"] == pytest.approx(1.001)
    assert payload["gpmd_residual_s"] == pytest.approx(0.601)


def test_gps_payload_records_a_null_residual_without_a_gpmd_track():
    payload = preproc.gps_payload([{"Main:Model": "Pixel 9 Pro"}], {"offset_s": 0.0, "r": 0.99, "ok": True})
    assert payload["gpmd_first_chunk_s"] is None
    assert payload["gpmd_residual_s"] is None


def test_gps_payload_marks_a_rejected_alignment_as_source_anchored():
    # Failed alignment takes the fix from the whole source, so it is flagged approximate
    payload = preproc.gps_payload(_DUMP, {"offset_s": 0.0, "r": 0.21, "ok": False})
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["gps_source_anchored"] is True


def test_gps_payload_reads_an_original_at_its_own_start():
    # An original's identity alignment puts the whole file in the cut, so its fix is exact
    payload = preproc.gps_payload(_DUMP, {"offset_s": 0.0, "r": 1.0, "ok": True})
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["gps_source_anchored"] is False


########
# Telemetry
########

# Three 1 s chunks of 3 IMU triplets each, a shortened stand-in for real ~202-triplet chunks
_IMU_DUMP = [
    {
        "Main:Model": "GoPro Max",
        "Doc1:SampleTime": 0.0,
        "Doc1:SampleDuration": 1.0,
        "Doc1:Accelerometer": "1 2 3 4 5 6 7 8 9",
        "Doc1:Gyroscope": "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
        "Doc2:SampleTime": 1.0,
        "Doc2:SampleDuration": 1.0,
        "Doc2:Accelerometer": "10 11 12 13 14 15 16 17 18",
        "Doc2:Gyroscope": "1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9",
        "Doc3:SampleTime": 2.0,
        "Doc3:SampleDuration": 1.0,
        "Doc3:Accelerometer": "19 20 21 22 23 24 25 26 27",
        "Doc3:Gyroscope": "2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9",
    }
]


def test_expand_gpmf_spreads_samples_across_the_chunk_duration():
    times, values = preproc.expand_gpmf(_IMU_DUMP, "Accelerometer", 3)
    assert len(times) == 9
    assert values[0] == (1.0, 2.0, 3.0)
    assert values[3] == (10.0, 11.0, 12.0)
    # Three samples spread over a 1 s chunk land at 0, 1/3, 2/3
    assert times[1] == pytest.approx(1 / 3)
    assert times[3] == pytest.approx(1.0)


def test_expand_gpmf_yields_samples_from_every_chunk():
    # All DocN groups share one object, so every chunk must contribute its own samples
    times, values = preproc.expand_gpmf(_IMU_DUMP, "Accelerometer", 3)
    assert values[0] == (1.0, 2.0, 3.0)
    assert values[3] == (10.0, 11.0, 12.0)
    assert values[6] == (19.0, 20.0, 21.0)
    assert times == sorted(times)
    assert times[-1] == pytest.approx(2 + 2 / 3)


def test_expand_gpmf_accepts_a_list_payload():
    # exiftool returns a list rather than a space-joined string for some tags
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:Accelerometer": [1, 2, 3]}]
    times, values = preproc.expand_gpmf(dump, "Accelerometer", 3)
    assert values == [(1.0, 2.0, 3.0)]
    assert times == [0.0]


def test_expand_gpmf_returns_empty_for_a_missing_key():
    assert preproc.expand_gpmf(_IMU_DUMP, "Gravity", 3) == ([], [])


def test_expand_gpmf_skips_a_binary_placeholder_chunk(caplog):
    # A non-numeric binary placeholder payload is logged and skipped like a ragged one
    placeholder = "(Binary data 10610 bytes, use -b option to extract)"
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:Accelerometer": placeholder}]
    with caplog.at_level("WARNING"):
        assert preproc.expand_gpmf(dump, "Accelerometer", 3) == ([], [])
    assert "Accelerometer" in caplog.text
    assert "Binary data" in caplog.text


def test_expand_gpmf_skips_a_ragged_chunk(caplog):
    # A truncated payload is dropped with a warning, so a wrong tag name is never silent
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:Accelerometer": "1 2 3 4"}]
    with caplog.at_level("WARNING"):
        assert preproc.expand_gpmf(dump, "Accelerometer", 3) == ([], [])
    assert "ragged Accelerometer chunk: 4 values for 3 components" in caplog.text


########
# expand_gpmf_parallel
########

# GPMF GPS as parallel single-value tags: one chunk fix plus an untimed DocN-M sub-sample
_GPS_DUMP = [
    {
        "Doc1:SampleTime": 0.0,
        "Doc1:SampleDuration": 1.0,
        "Doc1:GPSLatitude": 42.3532,
        "Doc1:GPSLongitude": -71.0659,
        "Doc1:GPSAltitude": 5.2,
        "Doc1:GPSSpeed": 1.5,
        "Doc1:GPSSpeed3D": 1.6,
        "Doc1-1:GPSLatitude": 42.3533,
        "Doc1-1:GPSLongitude": -71.0660,
        "Doc1-1:GPSAltitude": 5.3,
        "Doc1-1:GPSSpeed": 1.7,
        "Doc1-1:GPSSpeed3D": 1.8,
    }
]

# Two chunks carrying three fixes each, the real ~18 Hz-inside-1 Hz shape in miniature
_GPS_SUBSAMPLE_DUMP = [
    {
        "Doc1:SampleTime": 0.0,
        "Doc1:SampleDuration": 1.0,
        "Doc1:GPSLatitude": 42.3532,
        "Doc1:GPSLongitude": -71.0659,
        "Doc1-1:GPSLatitude": 42.35321,
        "Doc1-1:GPSLongitude": -71.06591,
        "Doc1-2:GPSLatitude": 42.35322,
        "Doc1-2:GPSLongitude": -71.06592,
        "Doc2:SampleTime": 1.0,
        "Doc2:SampleDuration": 1.0,
        "Doc2:GPSLatitude": 42.3533,
        "Doc2:GPSLongitude": -71.0660,
        "Doc2-1:GPSLatitude": 42.35331,
        "Doc2-1:GPSLongitude": -71.06601,
        "Doc2-2:GPSLatitude": 42.35332,
        "Doc2-2:GPSLongitude": -71.06602,
    }
]


def test_expand_gpmf_parallel_zips_one_row_per_fix():
    # Chunk 1 holds its own fix plus one sub-sample, so the two fixes split the 1 s window
    times, values = preproc.expand_gpmf_parallel(_GPS_DUMP, preproc._GPMF_GPS_TAGS)
    assert times == [0.0, 0.5]
    assert values[0] == (42.3532, -71.0659, 5.2, 1.5, 1.6)
    assert values[1] == (42.3533, -71.0660, 5.3, 1.7, 1.8)


def test_expand_gpmf_parallel_emits_every_sub_sample():
    # The ~18 Hz GPS stream inside a 1 Hz chunk yields one row per sub-sample
    times, values = preproc.expand_gpmf_parallel(_GPS_SUBSAMPLE_DUMP, preproc._GPMF_GPS_TAGS)
    assert len(times) == 6
    assert times == sorted(times)
    assert times[:3] == [pytest.approx(0.0), pytest.approx(1 / 3), pytest.approx(2 / 3)]
    assert [round(v[0], 5) for v in values] == [42.3532, 42.35321, 42.35322, 42.3533, 42.35331, 42.35332]


def test_expand_gpmf_parallel_tolerates_an_absent_component():
    # Firmware that omits GPSSpeed3D must still yield the other four, not an empty stream
    dump = [
        {"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:GPSLatitude": 42.3532, "Doc1:GPSLongitude": -71.0659}
    ]
    times, values = preproc.expand_gpmf_parallel(dump, preproc._GPMF_GPS_TAGS)
    assert times == [0.0]
    assert values == [(42.3532, -71.0659, None, None, None)]


def test_expand_gpmf_parallel_ignores_the_untimed_container_document():
    # The container carries GPSAltitude and one whole-file coordinate; it is not a sample
    dump = [{"Main:GPSLatitude": 42.3532, "Main:GPSLongitude": -71.0659, "Main:GPSAltitude": 5.2}]
    assert preproc.expand_gpmf_parallel(dump, preproc._GPMF_GPS_TAGS) == ([], [])


def test_telemetry_table_carries_both_time_axes():
    table = preproc.telemetry_table(_IMU_DUMP, 0.5, duration_s=1.0)
    columns = table.column_names
    assert "source_time" in columns and "edit_time" in columns and "in_edit" in columns
    assert "accl_x" in columns and "gyro_z" in columns
    edit_time = table.column("edit_time").to_pylist()
    source_time = table.column("source_time").to_pylist()
    # edit_time is source_time shifted by the solved offset
    assert edit_time[3] == pytest.approx(source_time[3] - 0.5)


def test_telemetry_table_marks_samples_outside_the_cut():
    table = preproc.telemetry_table(_IMU_DUMP, 0.5, duration_s=1.0)
    in_edit = table.column("in_edit").to_pylist()
    # The cut runs 0.5 s to 1.5 s in source time, so the first and last samples fall outside
    assert in_edit[0] is False
    assert in_edit[3] is True
    assert in_edit[-1] is False


def test_telemetry_table_nulls_edit_time_when_alignment_failed():
    table = preproc.telemetry_table(_IMU_DUMP, None, duration_s=1.0)
    assert set(table.column("edit_time").to_pylist()) == {None}
    assert set(table.column("in_edit").to_pylist()) == {None}


def test_telemetry_table_is_none_without_imu():
    assert preproc.telemetry_table([{"Main:Model": "Pixel 9 Pro"}], 0.0, 44.2) is None


def test_telemetry_table_populates_the_gps_columns():
    # GPS columns come from the parallel GPS tags and are populated per fix
    table = preproc.telemetry_table(_GPS_DUMP, 0.0, duration_s=1.0)
    assert {"gps_lat", "gps_lon", "gps_alt", "gps_speed2d", "gps_speed3d"} <= set(table.column_names)
    assert table.column("gps_lat").to_pylist() == [pytest.approx(42.3532), pytest.approx(42.3533)]
    assert table.column("gps_speed3d").to_pylist() == [pytest.approx(1.6), pytest.approx(1.8)]


def test_telemetry_table_carries_more_gps_rows_than_chunks():
    # Two chunks of three fixes each must produce six populated GPS rows, not two
    table = preproc.telemetry_table(_GPS_SUBSAMPLE_DUMP, 0.0, duration_s=2.0)
    populated = [value for value in table.column("gps_lat").to_pylist() if value is not None]
    assert len(populated) == 6
    assert table.num_rows == 6


def test_telemetry_table_never_reads_gps_as_a_ragged_wide_tag(caplog):
    # A wrong tag name announces itself as a ragged-chunk warning; there must be none
    with caplog.at_level("WARNING"):
        preproc.telemetry_table(_GPS_DUMP, 0.0, duration_s=1.0)
    assert "ragged" not in caplog.text


def test_telemetry_table_logs_the_row_count_and_fill_ratio(caplog):
    # Streams share no time axis, so the log reports fill ratios on the sparse union
    with caplog.at_level("INFO"):
        preproc.telemetry_table(_IMU_DUMP, 0.5, duration_s=1.0)
    assert "rows on the union time axis" in caplog.text
    assert "accl_x=100%" in caplog.text


def test_telemetry_table_columns_are_sparse_on_a_disjoint_axis():
    # Streams with different sample counts interleave, leaving each column null at the other's instants
    dump = [
        {
            "Doc1:SampleTime": 0.0,
            "Doc1:SampleDuration": 1.0,
            "Doc1:Accelerometer": "1 2 3 4 5 6",
            "Doc1:Gyroscope": "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
        }
    ]
    table = preproc.telemetry_table(dump, 0.0, duration_s=1.0)
    # 2 accelerometer samples at 0, 0.5 and 3 gyro samples at 0, 1/3, 2/3: union is 4 instants
    assert table.num_rows == 4
    assert table.column("accl_x").null_count == 2
    assert table.column("gyro_x").null_count == 1


def test_write_telemetry_names_the_file_after_the_video(tmp_path):
    table = preproc.telemetry_table(_IMU_DUMP, 0.5, duration_s=1.0)
    path = preproc.write_telemetry(table, tmp_path, Path("GH010234.mp4"))
    assert path.name == "GH010234_telemetry.parquet"
    assert path.is_file()


########
# Injection
########


def test_gpmd_command_trims_the_source_before_mapping_it():
    command = preproc.gpmd_command(
        Path("/out/GH010234.mp4"), Path("/src/GH010234.MP4"), 4.2, 131.4, 3, Path("/out/tmp.mp4")
    )
    # -ss and -t must precede the -i they apply to, or ffmpeg trims the wrong input
    assert command.index("-ss") < command.index("/src/GH010234.MP4")
    assert command[command.index("-ss") + 1] == "4.2"
    assert command[command.index("-t") + 1] == "131.4"


def test_gpmd_command_copies_every_curated_stream_and_only_the_gpmd_track():
    command = preproc.gpmd_command(
        Path("/out/GH010234.mp4"), Path("/src/GH010234.MP4"), 4.2, 131.4, 3, Path("/out/tmp.mp4")
    )
    assert "-map" in command and "0" in command
    assert "1:3" in command
    # -c copy keeps the edit's pixels untouched; -copy_unknown is what lets bin_data through
    assert "-c" in command and "copy" in command
    assert "-copy_unknown" in command


def test_gpmd_command_excludes_the_curated_datas_streams_after_mapping_them():
    # "-map -0:d" must follow "-map 0" to drop the export's unmuxable tmcd data stream
    command = preproc.gpmd_command(
        Path("/out/GH010234.mp4"), Path("/src/GH010234.MP4"), 4.2, 131.4, 3, Path("/out/tmp.mp4")
    )
    assert command.index("0") < command.index("-0:d")
    map_indices = [i for i, arg in enumerate(command) if arg == "-map"]
    assert command[map_indices[0] + 1] == "0"
    assert command[map_indices[1] + 1] == "-0:d"


def test_tag_command_overwrites_in_place():
    command = preproc.tag_command(Path("/out/GH010234.mp4"), {"Model": "GoPro Max"})
    assert "-overwrite_original" in command
    assert "-Model=GoPro Max" in command
    # QuickTimeUTC stops exiftool reinterpreting the capture time in local time
    assert "-api" in command and "QuickTimeUTC" in command


def test_tag_command_skips_empty_values():
    command = preproc.tag_command(Path("/out/GH010234.mp4"), {"Model": "GoPro Max", "SerialNumber": None})
    assert not any(arg.startswith("-SerialNumber") for arg in command)


def test_tag_command_writes_raw_numeric_values():
    # Tags arrive raw from exif_dump, so they must be written back with -n or exiftool drops them
    command = preproc.tag_command(Path("/out/GH010234.mp4"), {"GPSCoordinates": "42.3532 -71.0659 5.2"})
    assert "-n" in command
    assert "-GPSCoordinates=42.3532 -71.0659 5.2" in command


def test_tag_command_drops_tags_exiftool_cannot_write():
    # exiftool cannot write these two tags, so they are kept out of the call
    tags = {"Model": "GoPro Max", "LensProjection": "Fisheye", "ElectronicImageStabilization": 1}
    command = preproc.tag_command(Path("/out/GH010234.mp4"), tags)
    assert "-Model=GoPro Max" in command
    assert not any(arg.startswith("-LensProjection") for arg in command)
    assert not any(arg.startswith("-ElectronicImageStabilization") for arg in command)


def test_static_tags_still_carries_the_unwritable_tags():
    # Dropping them from the exiftool call must not drop them from the sidecar
    dump = [{"Main:LensProjection": "Fisheye", "Main:ElectronicImageStabilization": 1}]
    tags = preproc.static_tags(dump)
    assert tags["LensProjection"] == "Fisheye"
    assert tags["ElectronicImageStabilization"] == 1


def test_inject_warns_on_an_exiftool_warning_despite_a_zero_exit(tmp_path, monkeypatch, caplog):
    # exiftool exits 0 if any tag lands, so a dropped tag is detected from stderr warnings
    curated = tmp_path / "GH010234.mp4"
    curated.write_bytes(b"video")
    stderr = "Warning: Error converting value for ItemList:GPSCoordinates (PrintConvInv)\n"
    monkeypatch.setattr(
        preproc.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "1 image files updated\n", stderr),
    )
    with caplog.at_level("WARNING"):
        injected = preproc.inject(curated, tmp_path / "src.MP4", None, 1.0, {})

    assert injected is False
    assert "PrintConvInv" in caplog.text
    assert "GH010234.mp4" in caplog.text


def test_inject_raises_on_a_non_zero_exiftool_exit(tmp_path, monkeypatch):
    curated = tmp_path / "GH010234.mp4"
    curated.write_bytes(b"video")
    monkeypatch.setattr(
        preproc.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "Error: nothing to write\n"),
    )
    with pytest.raises(RuntimeError, match="static tags failed.*nothing to write"):
        preproc.inject(curated, tmp_path / "src.MP4", None, 1.0, {})


def test_inject_raises_on_a_failed_remux_and_removes_the_temp(tmp_path, monkeypatch):
    curated = tmp_path / "GH010234.mp4"
    curated.write_bytes(b"video")
    temp = tmp_path / "GH010234.inject.mp4"

    # ffmpeg leaves a partial temp behind and exits 1
    def fake_run(command, **kwargs):
        temp.write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 1, "", "Invalid data found\n")

    monkeypatch.setattr(preproc, "find_gpmd_index", lambda path: 3)
    monkeypatch.setattr(preproc.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="gpmd injection failed.*Invalid data"):
        preproc.inject(curated, tmp_path / "src.MP4", 0.0, 1.0, {})

    assert not temp.exists()
    assert curated.read_bytes() == b"video"


########
# Injection against the real binaries
########

_HAS_MEDIA_TOOLS = all(shutil.which(name) for name in ("ffmpeg", "ffprobe", "exiftool"))


def _synth_video(path, duration=1, extra_args=()):
    """Render a one-second mp4 with color bars and a tone via ffmpeg's lavfi sources."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=30:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            *extra_args,
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _synth_gpmd_source(path, duration=2):
    """Render an mp4 whose timecode track is patched from `tmcd` to `gpmd`, mimicking a GoPro."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name("tmcd_" + path.name)
    _synth_video(staging, duration=duration, extra_args=("-timecode", "00:00:00:00"))
    path.write_bytes(staging.read_bytes().replace(b"tmcd", b"gpmd"))
    staging.unlink()
    return path


def _stream_tags(path):
    """Return the codec_tag_string of every stream in `path`, via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_tag_string", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [s["codec_tag_string"] for s in json.loads(result.stdout)["streams"]]


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MEDIA_TOOLS, reason="needs ffmpeg, ffprobe and exiftool")
def test_inject_really_grafts_a_gpmd_track_onto_the_curated_video(tmp_path):
    # Run real ffmpeg on a curated file with a Resolve-style timecode track the muxer cannot map
    curated = _synth_video(tmp_path / "GH010234.mp4", extra_args=("-timecode", "00:00:00:00"))
    source = _synth_gpmd_source(tmp_path / "src" / "GH010234.MP4")
    before = curated.stat().st_size
    assert "gpmd" not in _stream_tags(curated)

    injected = preproc.inject(curated, source, 0.0, 1.0, {"Model": "GoPro Max"})

    assert injected is True
    assert "gpmd" in _stream_tags(curated)
    assert curated.stat().st_size > before
    # The remux left no temporary behind, under either spelling
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["GH010234.mp4"]


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MEDIA_TOOLS, reason="needs ffmpeg, ffprobe and exiftool")
def test_inject_really_writes_raw_gps_static_tags(tmp_path):
    # GPSCoordinates comes off exif_dump raw; without -n exiftool silently drops it and exits 0
    curated = _synth_video(tmp_path / "GH010234.mp4")
    source = _synth_gpmd_source(tmp_path / "src" / "GH010234.MP4")

    preproc.inject(
        curated,
        source,
        0.0,
        1.0,
        {"Model": "GoPro Max", "GPSCoordinates": "42.3532 -71.0659 5.2"},
    )

    result = subprocess.run(
        ["exiftool", "-n", "-json", "-GPSCoordinates", "-Model", "-Software", str(curated)],
        check=True,
        capture_output=True,
        text=True,
    )
    tags = json.loads(result.stdout)[0]
    assert tags["GPSCoordinates"] == "42.3532 -71.0659 5.2"
    assert tags["Model"] == "GoPro Max"
    assert tags["Software"] == preproc.PROVENANCE_TAG


def test_last_stderr_line_picks_the_last_meaningful_line():
    assert preproc._last_stderr_line("warning: x\nfatal: y\n\n") == "fatal: y"
    assert preproc._last_stderr_line("   \n\n") == "(no stderr)"
    assert preproc._last_stderr_line("") == "(no stderr)"


########
# CLI
########


def test_require_binaries_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: None if name == "exiftool" else "/usr/bin/" + name)
    with pytest.raises(SystemExit, match="exiftool"):
        preproc.require_binaries()


def test_require_binaries_passes_when_everything_is_present(monkeypatch):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    assert preproc.require_binaries() is None


def test_main_dry_run_copies_nothing(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    out = tmp_path / "curated"
    assert preproc.main([str(tree), "--output-root", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_main_index_only_rebuilds_without_touching_video(tmp_path, monkeypatch):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    out = tmp_path / "curated"
    _curated(out, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    assert preproc.main(["--output-root", str(out), "--index-only"]) == 0
    rows = list(csv.reader(io.StringIO((out / "index.csv").read_text())))
    assert rows[1][0] == "2026_07_22-splats-GH010234"


def test_main_only_filters_to_one_clip(tree, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    out = tmp_path / "curated"
    with caplog.at_level("INFO"):
        preproc.main([str(tree), "--output-root", str(out), "--only", "GH010228", "--dry-run"])
    assert "GH010228" in caplog.text
    assert "IMG_4085" not in caplog.text


########
# process_video
########


@pytest.fixture
def stub_externals(monkeypatch):
    """Replace the ffmpeg/exiftool/ffprobe calls so process_video runs without real media."""
    calls = {"inject": 0, "align": 0}

    def fake_inject(curated, source, offset_s, duration_s, tags):
        calls["inject"] += 1
        # Injection grows the curated file past its source: the idempotency trap itself
        curated.write_bytes(curated.read_bytes() + b"-gpmd-track")
        return True

    def fake_align(video, source, name, **kw):
        calls["align"] += 1
        return {"offset_s": 0.5, "r": 0.99, "ok": True}

    monkeypatch.setattr(preproc, "align", fake_align)
    monkeypatch.setattr(preproc, "exif_dump", lambda path: _IMU_DUMP)
    monkeypatch.setattr(preproc, "probe_duration", lambda path: 1.0)
    monkeypatch.setattr(preproc, "inject", fake_inject)
    return calls


NAME = "2026_07_22-splats-GH010234"


def _edit(tmp_path):
    """Write an export and the camera original it was cut from, and return both paths."""
    video = tmp_path / "GH010234.mp4"
    source = tmp_path / "src" / "GH010234.MP4"
    source.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"edited video")
    source.write_bytes(b"camera original")
    return video, source


def _original(tmp_path):
    """Write an un-edited camera original with no src/ counterpart."""
    video = tmp_path / "GH010234.mp4"
    video.write_bytes(b"camera original")
    return video


def test_process_video_writes_video_sidecar_and_telemetry(tmp_path, stub_externals):
    video, source = _edit(tmp_path)
    out = tmp_path / "curated"

    record = preproc.process_video(video, source, NAME, out, 0.95, force=False)

    folder = out / NAME
    assert record["status"] == "processed"
    assert record["imu"] is True
    assert record["injected"] is True
    assert (folder / "GH010234.mp4").is_file()
    assert (folder / "GH010234_metadata.json").is_file()
    assert (folder / "GH010234_telemetry.parquet").is_file()


def test_process_video_skips_a_second_run_over_an_injected_file(tmp_path, stub_externals):
    # Injection grows the curated mp4, so a rerun must skip rather than re-copy and re-inject
    video, source = _edit(tmp_path)
    out = tmp_path / "curated"
    preproc.process_video(video, source, NAME, out, 0.95, force=False)
    after_first = (out / NAME / "GH010234.mp4").read_bytes()

    record = preproc.process_video(video, source, NAME, out, 0.95, force=False)

    assert record == {"name": NAME, "status": "skipped"}
    assert stub_externals["inject"] == 1
    assert (out / NAME / "GH010234.mp4").read_bytes() == after_first


def test_process_video_reinjects_from_the_pristine_edit_under_force(tmp_path, stub_externals):
    video, source = _edit(tmp_path)
    out = tmp_path / "curated"
    preproc.process_video(video, source, NAME, out, 0.95, force=False)

    record = preproc.process_video(video, source, NAME, out, 0.95, force=True)

    assert record["status"] == "processed"
    assert stub_externals["inject"] == 2
    # The forced re-copy starts from the pristine edit, so the track is not injected twice
    assert (out / NAME / "GH010234.mp4").read_bytes() == b"edited video-gpmd-track"


def test_process_video_records_a_rejected_alignment(tmp_path, stub_externals, monkeypatch):
    monkeypatch.setattr(preproc, "align", lambda video, source, name, **kw: {"offset_s": 0.0, "r": 0.2, "ok": False})
    video, source = _edit(tmp_path)

    record = preproc.process_video(video, source, NAME, tmp_path / "curated", 0.95, force=False)

    assert record["aligned"] is False


def test_process_video_curates_an_original(tmp_path, stub_externals):
    # No src/ counterpart: the file is curated on its own, sidecars and all
    video = _original(tmp_path)
    out = tmp_path / "curated"

    record = preproc.process_video(video, None, NAME, out, 0.95, force=False)

    folder = out / NAME
    assert record["status"] == "processed"
    assert (folder / "GH010234.mp4").is_file()
    assert (folder / "GH010234_metadata.json").is_file()
    assert (folder / "GH010234_telemetry.parquet").is_file()


def test_process_video_never_aligns_or_injects_an_original(tmp_path, stub_externals):
    # An original has no cut to locate and a native gpmd track, so align and inject are skipped
    video = _original(tmp_path)

    record = preproc.process_video(video, None, NAME, tmp_path / "curated", 0.95, force=False)

    assert stub_externals["align"] == 0
    assert stub_externals["inject"] == 0
    assert record["injected"] is False


def test_process_video_leaves_an_originals_bytes_untouched(tmp_path, stub_externals):
    video = _original(tmp_path)
    out = tmp_path / "curated"

    preproc.process_video(video, None, NAME, out, 0.95, force=False)

    assert (out / NAME / "GH010234.mp4").read_bytes() == b"camera original"


def test_an_originals_sidecar_records_it_as_unedited_and_natively_timed(tmp_path, stub_externals):
    # gpmd_native marks an original's untouched track, which gpmd_injected alone would hide
    video = _original(tmp_path)
    out = tmp_path / "curated"

    preproc.process_video(video, None, NAME, out, 0.95, force=False)

    payload = json.loads((out / NAME / "GH010234_metadata.json").read_text())
    assert payload["edited"] is False
    assert payload["gpmd_native"] is True
    assert payload["gpmd_injected"] is False
    assert payload["source_path"] == str(video)
    assert payload["alignment"] == {"offset_s": 0.0, "r": 1.0, "ok": True}


def test_an_edits_sidecar_records_it_as_edited_against_its_source(tmp_path, stub_externals):
    video, source = _edit(tmp_path)
    out = tmp_path / "curated"

    preproc.process_video(video, source, NAME, out, 0.95, force=False)

    payload = json.loads((out / NAME / "GH010234_metadata.json").read_text())
    assert payload["edited"] is True
    assert payload["source_path"] == str(source)
    assert "gpmd_native" not in payload
    assert payload["alignment"]["offset_s"] == 0.5


def test_process_video_writes_a_sidecar_the_csv_index_can_read(tmp_path, stub_externals, monkeypatch):
    # End to end: the sidecar process_video writes must be readable by index_rows
    monkeypatch.setattr(preproc, "exif_dump", lambda path: _DUMP)
    video, source = _edit(tmp_path)
    out = tmp_path / "curated"

    preproc.process_video(video, source, NAME, out, 0.95, force=False)

    assert preproc.index_rows(out) == [(NAME, "42 deg 21' 11.52\" N, 71 deg 3' 57.24\" W")]


def test_process_video_refreshes_a_sidecar_whose_recorded_source_is_gone(tmp_path, stub_externals):
    # A pruned src/ copy leaves the fingerprint unchanged, so the stale sidecar must be refreshed
    video, source = _edit(tmp_path)
    out = tmp_path / "curated"
    preproc.process_video(video, source, NAME, out, 0.95, force=False)
    curated_bytes = (out / NAME / "GH010234.mp4").read_bytes()
    shutil.rmtree(source.parent)

    record = preproc.process_video(video, None, NAME, out, 0.95, force=False)

    payload = json.loads((out / NAME / "GH010234_metadata.json").read_text())
    assert record["status"] == "refreshed"
    assert payload["edited"] is False
    assert payload["source_path"] == str(video)
    # Provenance is corrected without re-copying or re-injecting byte-identical footage
    assert stub_externals["inject"] == 1
    assert (out / NAME / "GH010234.mp4").read_bytes() == curated_bytes


def test_process_video_refreshes_a_sidecar_written_before_the_edited_field(tmp_path, stub_externals):
    video, source = _edit(tmp_path)
    out = tmp_path / "curated"
    preproc.process_video(video, source, NAME, out, 0.95, force=False)
    path = out / NAME / "GH010234_metadata.json"
    payload = json.loads(path.read_text())
    del payload["edited"]
    path.write_text(json.dumps(payload))

    record = preproc.process_video(video, source, NAME, out, 0.95, force=False)

    assert record["status"] == "refreshed"
    assert json.loads(path.read_text())["edited"] is True


def test_process_video_leaves_a_current_sidecar_alone(tmp_path, stub_externals):
    video, source = _edit(tmp_path)
    out = tmp_path / "curated"
    preproc.process_video(video, source, NAME, out, 0.95, force=False)

    record = preproc.process_video(video, source, NAME, out, 0.95, force=False)

    assert record["status"] == "skipped"


def test_main_warns_about_every_unaligned_clip(tree, tmp_path, monkeypatch, caplog):
    # The summary's one job a human must not miss: which clips carry source-time telemetry
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        preproc,
        "process_video",
        lambda video, source, name, out, min_r, force: {
            "name": name,
            "status": "processed",
            "aligned": False,
            "r": 0.2,
            "imu": True,
            "injected": False,
        },
    )
    with caplog.at_level("WARNING"):
        preproc.main([str(tree), "--output-root", str(tmp_path / "out")])

    assert caplog.text.count("alignment rejected") == 5


def test_main_survives_a_clip_that_raises(tree, tmp_path, monkeypatch, caplog):
    # A clip that raises (e.g. no audio track) must not stop the index and summary
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)

    def explode_on_one(video, source, name, out, min_r, force):
        if "GH010228" in name:
            raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="Output file does not contain any stream")
        return {"name": name, "status": "processed", "aligned": True, "r": 0.99, "imu": True, "injected": True}

    monkeypatch.setattr(preproc, "process_video", explode_on_one)
    out = tmp_path / "curated"
    with caplog.at_level("INFO"):
        assert preproc.main([str(tree), "--output-root", str(out)]) == 0

    # The run finished: the index was still written and the survivors still counted
    assert (out / "index.csv").is_file()
    assert "4 processed, 0 refreshed, 0 skipped, 1 failed" in caplog.text
    # And the casualty is named individually, not buried in a count
    assert "processing failed, nothing curated: 2026_07_15-Goprosplat-GH010228" in caplog.text


def test_main_dry_run_forwards_the_flag_to_the_push_script(tree, tmp_path, monkeypatch):
    # --dry-run --push still calls push, forwarding the dry-run flag
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []
    monkeypatch.setattr(preproc, "push", lambda root, dry_run=False: calls.append(dry_run))

    preproc.main([str(tree), "--output-root", str(tmp_path / "out"), "--dry-run", "--push"])

    assert calls == [True]


def test_push_appends_dry_run_only_when_asked(monkeypatch):
    commands = []
    monkeypatch.setattr(preproc.subprocess, "run", lambda command, **kwargs: commands.append(command))

    preproc.push(Path("/out"), dry_run=True)
    preproc.push(Path("/out"), dry_run=False)

    assert commands[0] == [str(preproc.PUSH_SCRIPT), "--source", "/out", "--dry-run"]
    assert commands[1] == [str(preproc.PUSH_SCRIPT), "--source", "/out"]


def test_main_index_only_does_not_demand_the_media_tools(tmp_path, monkeypatch):
    # --index-only reads sidecars alone, so it must work on a machine with no ffmpeg
    monkeypatch.setattr(preproc.shutil, "which", lambda name: None)
    out = tmp_path / "curated"
    _curated(out, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})

    assert preproc.main(["--output-root", str(out), "--index-only"]) == 0
    assert (out / "index.csv").is_file()


def test_main_says_when_only_matched_nothing(tree, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    with caplog.at_level("WARNING"):
        preproc.main([str(tree), "--output-root", str(tmp_path / "out"), "--only", "nope"])

    assert "no videos matched" in caplog.text


########
# Pruning redundant src/ copies
########


def _duplicate(root, folder="2026-03-27/videos_for_splats", stem="PXL_1", payload=b"identical bytes"):
    """Write a video and a byte-for-byte identical copy of it inside src/."""
    parent = root.joinpath(*folder.split("/"))
    video = parent / f"{stem}.mp4"
    source = parent / "src" / f"{stem}.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(payload)
    source.write_bytes(payload)
    return video, source


def test_is_redundant_sees_a_byte_identical_copy(tmp_path):
    video, source = _duplicate(tmp_path / "gdrive-src")
    assert preproc.is_redundant(video, source) is True


def test_is_redundant_rejects_a_same_size_re_encode(tmp_path):
    # An untrimmed re-encode can match in size but differ in bytes, so it is not redundant
    video, source = _duplicate(tmp_path / "gdrive-src")
    source.write_bytes(b"re-encoded byte")
    assert video.stat().st_size == source.stat().st_size
    assert preproc.is_redundant(video, source) is False


def test_is_redundant_rejects_a_trimmed_export(tmp_path):
    video, source = _duplicate(tmp_path / "gdrive-src")
    video.write_bytes(b"short cut")
    assert preproc.is_redundant(video, source) is False


def test_prune_src_reports_both_paths_without_deleting(tmp_path):
    root = tmp_path / "gdrive-src"
    video, source = _duplicate(root)
    removed = preproc.prune_src(root, apply=False)
    assert removed == 0
    assert source.is_file()
    assert video.is_file()


def test_prune_src_apply_deletes_the_copy_and_keeps_the_video(tmp_path):
    root = tmp_path / "gdrive-src"
    video, source = _duplicate(root)

    removed = preproc.prune_src(root, apply=True)

    assert removed == 1
    assert not source.exists()
    assert video.read_bytes() == b"identical bytes"


def test_prune_src_apply_removes_a_src_left_empty(tmp_path):
    root = tmp_path / "gdrive-src"
    _, source = _duplicate(root)

    preproc.prune_src(root, apply=True)

    assert not source.parent.exists()


def test_prune_src_keeps_a_src_that_still_holds_an_orphan(tmp_path):
    root = tmp_path / "gdrive-src"
    _, source = _duplicate(root)
    orphan = source.parent / "GH010233.MP4"
    orphan.write_bytes(b"an original whose export was never made")

    preproc.prune_src(root, apply=True)

    assert not source.exists()
    assert orphan.is_file()


def test_prune_src_leaves_a_real_export_alone(tmp_path):
    root = tmp_path / "gdrive-src"
    video, source = _duplicate(root)
    video.write_bytes(b"a genuinely different, trimmed export")

    removed = preproc.prune_src(root, apply=True)

    assert removed == 0
    assert source.is_file()


def test_prune_src_honours_only(tmp_path):
    root = tmp_path / "gdrive-src"
    _, kept = _duplicate(root, folder="2026-03-27/videos_for_splats", stem="PXL_1")
    _, pruned = _duplicate(root, folder="2026-07-22/splats", stem="GH010234")

    preproc.prune_src(root, only="2026_07_22", apply=True)

    assert not pruned.exists()
    assert kept.is_file()


def test_prune_src_reverifies_immediately_before_deleting(tmp_path, monkeypatch):
    # The Drive-synced file may change after the scan, so it is re-checked before unlinking
    root = tmp_path / "gdrive-src"
    _, source = _duplicate(root)
    answers = iter([True, False])
    monkeypatch.setattr(preproc, "is_redundant", lambda video, src: next(answers))

    removed = preproc.prune_src(root, apply=True)

    assert removed == 0
    assert source.is_file()


def test_main_prune_src_is_a_report_unless_applied(tmp_path, monkeypatch):
    # Pruning reads sidecar-free bytes only, so it must work on a machine with no ffmpeg
    monkeypatch.setattr(preproc.shutil, "which", lambda name: None)
    root = tmp_path / "gdrive-src"
    _, source = _duplicate(root)

    assert preproc.main([str(root), "--prune-src"]) == 0
    assert source.is_file()

    assert preproc.main([str(root), "--prune-src", "--apply"]) == 0
    assert not source.exists()
