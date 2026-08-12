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
    body = rng.standard_normal(preproc.AUDIO_RATE * 3).astype(np.float32)
    head = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    tail = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    source = np.concatenate([head, body, tail])

    result = preproc.solve_offset(body, source)
    assert result.ok is True
    assert result.r == pytest.approx(1.0, abs=1e-4)
    # Within one audio sample of the true 1.0 s offset
    assert abs(result.offset_s - 1.0) <= 1.0 / preproc.AUDIO_RATE


def test_solve_offset_is_unaffected_by_a_gain_change():
    # Normalized correlation is scale-free, so a level difference must not lower r
    rng = _rng()
    body = rng.standard_normal(preproc.AUDIO_RATE * 2).astype(np.float32)
    source = np.concatenate([rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32), body])

    result = preproc.solve_offset(body * 0.25, source)
    assert result.ok is True
    assert result.r == pytest.approx(1.0, abs=1e-4)


def test_solve_offset_rejects_uncorrelated_audio():
    rng = _rng()
    edit = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    source = rng.standard_normal(preproc.AUDIO_RATE * 5).astype(np.float32)

    result = preproc.solve_offset(edit, source)
    assert result.ok is False
    assert result.r < preproc.DEFAULT_ALIGN_MIN_R


def test_solve_offset_rejects_an_edit_longer_than_its_source():
    rng = _rng()
    edit = rng.standard_normal(preproc.AUDIO_RATE * 5).astype(np.float32)
    source = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)

    result = preproc.solve_offset(edit, source)
    assert result.ok is False
    assert result.offset_s == 0.0


def test_solve_offset_rejects_silence():
    # A constant signal has zero variance, so correlation is undefined rather than perfect
    source = np.zeros(preproc.AUDIO_RATE * 3, dtype=np.float32)
    result = preproc.solve_offset(np.zeros(preproc.AUDIO_RATE, dtype=np.float32), source)
    assert result.ok is False


def test_solve_offset_survives_a_silent_source_window():
    # A source that opens with digital silence drives the denominator to zero at lag 0.
    # The guard must yield 0 there rather than a nan, which would otherwise win the argmax
    # and return a confidently wrong offset.
    rng = _rng()
    body = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    source = np.concatenate([np.zeros(preproc.AUDIO_RATE, dtype=np.float32), body])

    result = preproc.solve_offset(body, source)
    assert result.ok is True
    assert abs(result.offset_s - 1.0) <= 1.0 / preproc.AUDIO_RATE
    assert np.isfinite(result.r)


def test_solve_offset_threshold_is_overridable():
    rng = _rng()
    edit = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    source = rng.standard_normal(preproc.AUDIO_RATE * 5).astype(np.float32)

    assert preproc.solve_offset(edit, source, min_r=0.0).ok is True


def test_solve_offset_recovers_a_lag_that_overruns_the_source_end():
    # The GH010218 regression: Resolve's AAC re-encode leaves the decoded edit longer than the
    # region it was cut from, so the true lag sits just above len(source) - len(edit). Before
    # the tolerance existed that lag was unreachable and the clip scored 0.09 instead of 1.0.
    rng = _rng()
    body = rng.standard_normal(preproc.AUDIO_RATE * 3).astype(np.float32)
    head = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    source = np.concatenate([head, body])
    # A short tail with no counterpart in the source, standing in for the encoder's padding
    edit = np.concatenate([body, rng.standard_normal(400).astype(np.float32)])
    # The true lag exceeds what a strictly-fitting search could reach
    assert preproc.AUDIO_RATE > len(source) - len(edit)

    result = preproc.solve_offset(edit, source)
    assert result.ok is True
    assert abs(result.offset_s - 1.0) <= 1.0 / preproc.AUDIO_RATE


def test_solve_offset_rejects_an_edit_longer_than_the_source_plus_tolerance():
    # The tolerance widens the search; it must not become a way to accept nonsense
    rng = _rng()
    edit = rng.standard_normal(preproc.AUDIO_RATE * 5).astype(np.float32)
    source = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)

    result = preproc.solve_offset(edit, source)
    assert result.ok is False
    assert result.offset_s == 0.0


########
# Metadata extraction
########

# One 1 Hz GPMF chunk per document, shaped the way exiftool -ee -json -G3 -n returns them
_DUMP = [
    {
        "Main:Make": "GoPro",
        "Main:Model": "GoPro Max",
        "Main:SerialNumber": "C123456789",
        "Main:FirmwareVersion": "H19.03.02.00",
        "Main:CreateDate": "2026:07:22 14:31:08",
        "Main:FieldOfView": "Wide",
    },
    {
        "Doc1:SampleTime": 0.0,
        "Doc1:SampleDuration": 1.001,
        "Doc1:GPSLatitude": 0.0,
        "Doc1:GPSLongitude": 0.0,
        "Doc1:GPSDateTime": "2026:07:22 14:31:08",
    },
    {
        "Doc2:SampleTime": 1.001,
        "Doc2:SampleDuration": 1.001,
        "Doc2:GPSLatitude": 42.3532,
        "Doc2:GPSLongitude": -71.0659,
        "Doc2:GPSDateTime": "2026:07:22 14:31:09",
    },
]


def test_static_tags_strips_the_group_prefix():
    tags = preproc.static_tags(_DUMP)
    assert tags["Model"] == "GoPro Max"
    assert tags["SerialNumber"] == "C123456789"
    assert tags["CreateDate"] == "2026:07:22 14:31:08"


def test_first_fix_skips_the_unlocked_zero_zero_sample():
    # A GoPro emits 0,0 before satellite lock; treating that as a fix would put every early
    # clip in the Gulf of Guinea
    fix = preproc.first_fix(_DUMP)
    assert fix["latitude"] == pytest.approx(42.3532)
    assert fix["longitude"] == pytest.approx(-71.0659)
    assert fix["source_time"] == pytest.approx(1.001)


def test_first_fix_returns_none_when_nothing_locked():
    assert preproc.first_fix([{"Doc1:SampleTime": 0.0, "Doc1:GPSLatitude": 0.0, "Doc1:GPSLongitude": 0.0}]) is None


def test_first_fix_reads_a_container_location_when_there_is_no_track():
    # Pixel and iPhone carry a single point and no GPS track at all
    dump = [{"Main:GPSLatitude": 42.3532, "Main:GPSLongitude": -71.0659}]
    fix = preproc.first_fix(dump)
    assert fix["latitude"] == pytest.approx(42.3532)
    assert fix["source_time"] == 0.0


def test_first_fix_honours_the_trim_window():
    # A fix before the cut starts is not where this clip was shot
    fix = preproc.first_fix(_DUMP, start_s=1.5)
    assert fix is None


def test_build_payload_satisfies_the_index_contract():
    pair = preproc.Pair(Path("/x/GH010234.mp4"), Path("/x/src/GH010234.MP4"), "2026_07_22-splats-GH010234")
    payload = preproc.build_payload(
        pair,
        _DUMP,
        preproc.Alignment(4.2, 0.997, True),
        duration_s=131.4,
        unique_id="2026_07_22-splats-GH010234",
        has_imu=True,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    assert payload["unique_id"] == "2026_07_22-splats-GH010234"
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["alignment"] == {"offset_s": 4.2, "r": 0.997, "ok": True}
    assert payload["source"] == {"size_bytes": 5, "mtime": 1.0}
    assert payload["has_imu"] is True
    # Cuts plus colour correction only, so lens geometry still describes the exported pixels
    assert payload["intrinsics"]["valid_for_edit"] is True
    # No locked fix at or after 4.2 s in _DUMP, so this exercises the whole-source fallback
    assert payload["gps_source_anchored"] is True


def test_build_payload_records_a_missing_fix_as_null():
    pair = preproc.Pair(Path("/x/PXL.mp4"), Path("/x/src/PXL.mp4"), "2026_06_29-Phone-PXL")
    payload = preproc.build_payload(
        pair,
        [{"Main:Model": "Pixel 9 Pro"}],
        preproc.Alignment(0.0, 0.99, True),
        duration_s=44.2,
        unique_id="2026_06_29-Phone-PXL",
        has_imu=False,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    assert payload["gps"] is None
    assert payload["has_imu"] is False
    # No fix anywhere, so the flag must not claim an approximate fix exists
    assert payload["gps_source_anchored"] is False


def test_build_payload_marks_a_fix_inside_the_cut_as_exact():
    # The cut starts at 0.5 s and _DUMP has a locked fix at 1.001 s, so the fix comes
    # from inside the trim and must not be labelled source-anchored
    pair = preproc.Pair(Path("/x/GH010234.mp4"), Path("/x/src/GH010234.MP4"), "2026_07_22-splats-GH010234")
    payload = preproc.build_payload(
        pair,
        _DUMP,
        preproc.Alignment(0.5, 0.997, True),
        duration_s=131.4,
        unique_id="2026_07_22-splats-GH010234",
        has_imu=True,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["gps_source_anchored"] is False


def test_build_payload_does_not_anchor_a_container_only_fix():
    # A phone clip has one untimed whole-file coordinate reported at 0, so any non-zero offset
    # skips it and falls through to the whole-source search. That is not a lost timed fix, and
    # labelling it source-anchored stamped the flag on every phone clip in the tree.
    pair = preproc.Pair(Path("/x/PXL.mp4"), Path("/x/src/PXL.mp4"), "2026_06_29-Phone-PXL")
    payload = preproc.build_payload(
        pair,
        [{"Main:Model": "Pixel 9 Pro", "Main:GPSLatitude": 42.3532, "Main:GPSLongitude": -71.0659}],
        preproc.Alignment(4.2, 0.99, True),
        duration_s=44.2,
        unique_id="2026_06_29-Phone-PXL",
        has_imu=False,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["gps_source_anchored"] is False


def test_build_payload_records_the_gpmd_chunk_residual():
    # GPMF chunks are 1 Hz, so the injected track starts on the first boundary at or after the
    # cut. The residual is what a consumer needs to line the injected track up with the Parquet.
    pair = preproc.Pair(Path("/x/GH010234.mp4"), Path("/x/src/GH010234.MP4"), "2026_07_22-splats-GH010234")
    payload = preproc.build_payload(
        pair,
        _DUMP,
        preproc.Alignment(0.4, 0.997, True),
        duration_s=131.4,
        unique_id="2026_07_22-splats-GH010234",
        has_imu=True,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    # _DUMP has chunks at 0.0 and 1.001; the first at or after 0.4 is 1.001
    assert payload["gpmd_first_chunk_s"] == pytest.approx(1.001)
    assert payload["gpmd_residual_s"] == pytest.approx(0.601)


def test_build_payload_records_a_null_residual_without_a_gpmd_track():
    pair = preproc.Pair(Path("/x/PXL.mp4"), Path("/x/src/PXL.mp4"), "2026_06_29-Phone-PXL")
    payload = preproc.build_payload(
        pair,
        [{"Main:Model": "Pixel 9 Pro"}],
        preproc.Alignment(0.0, 0.99, True),
        duration_s=44.2,
        unique_id="2026_06_29-Phone-PXL",
        has_imu=False,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    assert payload["gpmd_first_chunk_s"] is None
    assert payload["gpmd_residual_s"] is None


def test_build_payload_reaches_the_csv_index_through_the_real_sidecar(tmp_path):
    # The seam every other index test skips by hand-building its sidecar: rename `latitude` in
    # build_payload and those tests all stay green while every GPS cell in the CSV goes blank
    unique_id = "2026_07_22-splats-GH010234"
    video = Path("GH010234.mp4")
    pair = preproc.Pair(Path("/x/GH010234.mp4"), Path("/x/src/GH010234.MP4"), unique_id)
    payload = preproc.build_payload(
        pair,
        _DUMP,
        preproc.Alignment(0.5, 0.997, True),
        duration_s=131.4,
        unique_id=unique_id,
        has_imu=True,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    preproc.write_metadata(tmp_path / unique_id, video, payload)

    rows = preproc.index_rows(tmp_path)

    assert rows == [(unique_id, "42 deg 21' 11.52\" N, 71 deg 3' 57.24\" W")]


def test_build_payload_marks_a_rejected_alignment_as_source_anchored():
    # Alignment failed, so the trim window is unknown and the fix is taken from the
    # whole source; the flag records that the value is approximate
    pair = preproc.Pair(Path("/x/GH010234.mp4"), Path("/x/src/GH010234.MP4"), "2026_07_22-splats-GH010234")
    payload = preproc.build_payload(
        pair,
        _DUMP,
        preproc.Alignment(0.0, 0.21, False),
        duration_s=131.4,
        unique_id="2026_07_22-splats-GH010234",
        has_imu=True,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["gps_source_anchored"] is True
    assert payload["alignment"]["ok"] is False


########
# Telemetry
########

# Two 1 Hz chunks, each holding 3 accelerometer triplets — the shape exiftool returns
_IMU_DUMP = [
    {
        "Doc1:SampleTime": 0.0,
        "Doc1:SampleDuration": 1.0,
        "Doc1:Accelerometer": "1 2 3 4 5 6 7 8 9",
        "Doc1:Gyroscope": "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
    },
    {
        "Doc2:SampleTime": 1.0,
        "Doc2:SampleDuration": 1.0,
        "Doc2:Accelerometer": "10 11 12 13 14 15 16 17 18",
        "Doc2:Gyroscope": "1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9",
    },
]


def test_expand_gpmf_spreads_samples_across_the_chunk_duration():
    times, values = preproc.expand_gpmf(_IMU_DUMP, "Accelerometer", 3)
    assert len(times) == 6
    assert values[0] == (1.0, 2.0, 3.0)
    assert values[3] == (10.0, 11.0, 12.0)
    # Three samples spread over a 1 s chunk land at 0, 1/3, 2/3
    assert times[1] == pytest.approx(1 / 3)
    assert times[3] == pytest.approx(1.0)


def test_expand_gpmf_accepts_a_list_payload():
    # exiftool returns a list rather than a space-joined string for some tags
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:Accelerometer": [1, 2, 3]}]
    times, values = preproc.expand_gpmf(dump, "Accelerometer", 3)
    assert values == [(1.0, 2.0, 3.0)]
    assert times == [0.0]


def test_expand_gpmf_returns_empty_for_a_missing_key():
    assert preproc.expand_gpmf(_IMU_DUMP, "Gravity", 3) == ([], [])


def test_expand_gpmf_skips_a_ragged_chunk(caplog):
    # A truncated payload cannot be split into whole triplets; dropping it beats guessing.
    # The warning is the load-bearing half: a wrong tag name shows up here and nowhere else,
    # so an empty return with no warning would look identical to a clip that has no IMU.
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:Accelerometer": "1 2 3 4"}]
    with caplog.at_level("WARNING"):
        assert preproc.expand_gpmf(dump, "Accelerometer", 3) == ([], [])
    assert "ragged Accelerometer chunk: 4 values for 3 components" in caplog.text


########
# expand_gpmf_parallel
########

# exiftool emits the GPMF GPS stream as one document per fix with the components side by side,
# not as one wide tag — the shape the old 5-wide GPSTrack read could never match
_GPS_DUMP = [
    {
        "Doc1:SampleTime": 0.0,
        "Doc1:GPSLatitude": 42.3532,
        "Doc1:GPSLongitude": -71.0659,
        "Doc1:GPSAltitude": 5.2,
        "Doc1:GPSSpeed": 1.5,
        "Doc1:GPSSpeed3D": 1.6,
    },
    {
        "Doc2:SampleTime": 0.5,
        "Doc2:GPSLatitude": 42.3533,
        "Doc2:GPSLongitude": -71.0660,
        "Doc2:GPSAltitude": 5.3,
        "Doc2:GPSSpeed": 1.7,
        "Doc2:GPSSpeed3D": 1.8,
    },
]


def test_expand_gpmf_parallel_zips_one_row_per_document():
    times, values = preproc.expand_gpmf_parallel(_GPS_DUMP, preproc._GPMF_GPS_TAGS)
    assert times == [0.0, 0.5]
    assert values[0] == (42.3532, -71.0659, 5.2, 1.5, 1.6)
    assert values[1] == (42.3533, -71.0660, 5.3, 1.7, 1.8)


def test_expand_gpmf_parallel_tolerates_an_absent_component():
    # Firmware that omits GPSSpeed3D must still yield the other four, not an empty stream
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:GPSLatitude": 42.3532, "Doc1:GPSLongitude": -71.0659}]
    times, values = preproc.expand_gpmf_parallel(dump, preproc._GPMF_GPS_TAGS)
    assert times == [0.0]
    assert values == [(42.3532, -71.0659, None, None, None)]


def test_expand_gpmf_parallel_ignores_the_untimed_container_document():
    # The container carries GPSAltitude and one whole-file coordinate; it is not a sample
    dump = [{"Main:GPSLatitude": 42.3532, "Main:GPSLongitude": -71.0659, "Main:GPSAltitude": 5.2}]
    assert preproc.expand_gpmf_parallel(dump, preproc._GPMF_GPS_TAGS) == ([], [])


def test_telemetry_table_carries_both_time_axes():
    table = preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.5, 0.99, True), duration_s=1.0)
    columns = table.column_names
    assert "source_time" in columns and "edit_time" in columns and "in_edit" in columns
    assert "accl_x" in columns and "gyro_z" in columns
    edit_time = table.column("edit_time").to_pylist()
    source_time = table.column("source_time").to_pylist()
    # edit_time is source_time shifted by the solved offset
    assert edit_time[3] == pytest.approx(source_time[3] - 0.5)


def test_telemetry_table_marks_samples_outside_the_cut():
    table = preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.5, 0.99, True), duration_s=1.0)
    in_edit = table.column("in_edit").to_pylist()
    # The cut runs 0.5 s to 1.5 s in source time, so the first and last samples fall outside
    assert in_edit[0] is False
    assert in_edit[3] is True
    assert in_edit[-1] is False


def test_telemetry_table_nulls_edit_time_when_alignment_failed():
    table = preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.0, 0.2, False), duration_s=1.0)
    assert set(table.column("edit_time").to_pylist()) == {None}
    assert set(table.column("in_edit").to_pylist()) == {None}


def test_telemetry_table_is_none_without_imu():
    assert preproc.telemetry_table([{"Main:Model": "Pixel 9 Pro"}], preproc.Alignment(0.0, 0.99, True), 44.2) is None


def test_telemetry_table_populates_the_gps_columns():
    # The regression this guards: gps was read as one 5-wide "GPSTrack", which is exiftool's
    # scalar heading, so every chunk was skipped as ragged and gps_* was always null
    table = preproc.telemetry_table(_GPS_DUMP, preproc.Alignment(0.0, 0.99, True), duration_s=1.0)
    assert {"gps_lat", "gps_lon", "gps_alt", "gps_speed2d", "gps_speed3d"} <= set(table.column_names)
    assert table.column("gps_lat").to_pylist() == [pytest.approx(42.3532), pytest.approx(42.3533)]
    assert table.column("gps_speed3d").to_pylist() == [pytest.approx(1.6), pytest.approx(1.8)]


def test_telemetry_table_never_reads_gps_as_a_ragged_wide_tag(caplog):
    # A wrong tag name announces itself as a ragged-chunk warning; there must be none
    with caplog.at_level("WARNING"):
        preproc.telemetry_table(_GPS_DUMP, preproc.Alignment(0.0, 0.99, True), duration_s=1.0)
    assert "ragged" not in caplog.text


def test_telemetry_table_logs_the_row_count_and_fill_ratio(caplog):
    # The streams do not share a time axis, so the union is sparse; the log is how a reader
    # learns that a half-null column is expected rather than a bug
    with caplog.at_level("INFO"):
        preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.5, 0.99, True), duration_s=1.0)
    assert "rows on the union time axis" in caplog.text
    assert "accl_x=100%" in caplog.text


def test_telemetry_table_columns_are_sparse_on_a_disjoint_axis():
    # Two streams whose chunks hold different sample counts land on different instants, so
    # the union carries both and each column is null wherever the other stream sampled
    dump = [
        {
            "Doc1:SampleTime": 0.0,
            "Doc1:SampleDuration": 1.0,
            "Doc1:Accelerometer": "1 2 3 4 5 6",
            "Doc1:Gyroscope": "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
        }
    ]
    table = preproc.telemetry_table(dump, preproc.Alignment(0.0, 0.99, True), duration_s=1.0)
    # 2 accelerometer samples at 0, 0.5 and 3 gyro samples at 0, 1/3, 2/3: union is 4 instants
    assert table.num_rows == 4
    assert table.column("accl_x").null_count == 2
    assert table.column("gyro_x").null_count == 1


def test_write_telemetry_names_the_file_after_the_video(tmp_path):
    table = preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.5, 0.99, True), duration_s=1.0)
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
    # exif_dump reads with -n, so GPSCoordinates arrives already raw. Writing it back without
    # -n runs PrintConvInv over a raw value: exiftool warns, drops the tag, and still exits 0
    command = preproc.tag_command(Path("/out/GH010234.mp4"), {"GPSCoordinates": "42.3532 -71.0659 5.2"})
    assert "-n" in command
    assert "-GPSCoordinates=42.3532 -71.0659 5.2" in command


def test_tag_command_drops_tags_exiftool_cannot_write():
    # exiftool answers "Sorry, <tag> is not writable" for these two. They stay in the JSON
    # sidecar via static_tags, but passing them to exiftool only buys a warning.
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
    # The branch this replaces was dead: exiftool exits 0 whenever any tag in the call lands,
    # so a dropped tag shows up only in stderr. Asserting on the returncode would prove nothing.
    curated = tmp_path / "GH010234.mp4"
    curated.write_bytes(b"video")
    stderr = "Warning: Error converting value for ItemList:GPSCoordinates (PrintConvInv)\n"
    monkeypatch.setattr(
        preproc.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "1 image files updated\n", stderr),
    )
    with caplog.at_level("WARNING"):
        injected = preproc.inject(curated, tmp_path / "src.MP4", preproc.Alignment(0.0, 0.2, False), 1.0, {})

    assert injected is False
    assert "PrintConvInv" in caplog.text
    assert "GH010234.mp4" in caplog.text


def test_inject_logs_a_non_zero_exiftool_exit_as_an_error(tmp_path, monkeypatch, caplog):
    curated = tmp_path / "GH010234.mp4"
    curated.write_bytes(b"video")
    monkeypatch.setattr(
        preproc.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "Error: nothing to write\n"),
    )
    with caplog.at_level("WARNING"):
        preproc.inject(curated, tmp_path / "src.MP4", preproc.Alignment(0.0, 0.2, False), 1.0, {})

    assert "static tags failed" in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)


########
# Injection against the real binaries
########

_HAS_MEDIA_TOOLS = all(shutil.which(name) for name in ("ffmpeg", "ffprobe", "exiftool"))


def _synth_video(path, duration=1, extra_args=()):
    """Render a one-second mp4 with colour bars and a tone via ffmpeg's lavfi sources."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            *extra_args, str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _synth_gpmd_source(path, duration=2):
    """Render an mp4 carrying a data track tagged `gpmd`, the way a GoPro original does.

    ffmpeg cannot author a bin_data stream from scratch, so a timecode track is rendered and
    its four-character format code is patched from `tmcd` to `gpmd`. ffprobe then reports the
    stream exactly as it reports a real GoPro's GPMF track, which is all find_gpmd_index reads.
    """
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
    # The four argv-shape tests above never ran ffmpeg, which is how a temp file named
    # "GH010234.mp4.inject" survived: ffmpeg cannot infer a muxer from that suffix and exits 1
    # with "Unable to find a suitable output format", so every GoPro clip took the failure path.
    curated = _synth_video(tmp_path / "GH010234.mp4")
    source = _synth_gpmd_source(tmp_path / "src" / "GH010234.MP4")
    before = curated.stat().st_size
    assert "gpmd" not in _stream_tags(curated)

    injected = preproc.inject(
        curated, source, preproc.Alignment(0.0, 0.99, True), 1.0, {"Model": "GoPro Max"}
    )

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
        preproc.Alignment(0.0, 0.99, True),
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
    assert preproc.main(["--source-root", str(tree), "--output-root", str(out), "--dry-run"]) == 0
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
        preproc.main(["--source-root", str(tree), "--output-root", str(out), "--only", "GH010228", "--dry-run"])
    assert "GH010228" in caplog.text
    assert "IMG_4085" not in caplog.text


########
# process_pair
########


@pytest.fixture
def stub_externals(monkeypatch):
    """Replace the ffmpeg/exiftool/ffprobe calls so process_pair runs without real media."""
    calls = {"inject": 0}

    def fake_inject(curated, source, alignment, duration_s, tags):
        calls["inject"] += 1
        # Injection grows the curated file past its source: the idempotency trap itself
        curated.write_bytes(curated.read_bytes() + b"-gpmd-track")
        return True

    monkeypatch.setattr(preproc, "align", lambda pair, **kw: preproc.Alignment(0.5, 0.99, True))
    monkeypatch.setattr(preproc, "exif_dump", lambda path: _IMU_DUMP)
    monkeypatch.setattr(preproc, "probe_duration", lambda path: 1.0)
    monkeypatch.setattr(preproc, "inject", fake_inject)
    return calls


def _pair(tmp_path):
    """Build a Pair whose edit and camera original both exist on disk."""
    edit = tmp_path / "GH010234.mp4"
    source = tmp_path / "src" / "GH010234.MP4"
    source.parent.mkdir(parents=True, exist_ok=True)
    edit.write_bytes(b"edited video")
    source.write_bytes(b"camera original")
    return preproc.Pair(edit, source, "2026_07_22-splats-GH010234")


def test_process_pair_writes_video_sidecar_and_telemetry(tmp_path, stub_externals):
    pair = _pair(tmp_path)
    out = tmp_path / "curated"

    record = preproc.process_pair(pair, out, preproc.DEFAULT_ALIGN_MIN_R, force=False)

    folder = out / pair.name
    assert record["status"] == "processed"
    assert record["imu"] is True
    assert record["injected"] is True
    assert (folder / "GH010234.mp4").is_file()
    assert (folder / "GH010234_metadata.json").is_file()
    assert (folder / "GH010234_telemetry.parquet").is_file()


def test_process_pair_skips_a_second_run_over_an_injected_file(tmp_path, stub_externals):
    # The trap this pipeline exists to avoid: injection left the curated mp4 larger than
    # its source, so a destination-size check would re-copy and re-inject forever
    pair = _pair(tmp_path)
    out = tmp_path / "curated"
    preproc.process_pair(pair, out, preproc.DEFAULT_ALIGN_MIN_R, force=False)
    after_first = (out / pair.name / "GH010234.mp4").read_bytes()

    record = preproc.process_pair(pair, out, preproc.DEFAULT_ALIGN_MIN_R, force=False)

    assert record == {"name": pair.name, "status": "skipped"}
    assert stub_externals["inject"] == 1
    assert (out / pair.name / "GH010234.mp4").read_bytes() == after_first


def test_process_pair_reinjects_from_the_pristine_edit_under_force(tmp_path, stub_externals):
    pair = _pair(tmp_path)
    out = tmp_path / "curated"
    preproc.process_pair(pair, out, preproc.DEFAULT_ALIGN_MIN_R, force=False)

    record = preproc.process_pair(pair, out, preproc.DEFAULT_ALIGN_MIN_R, force=True)

    assert record["status"] == "processed"
    assert stub_externals["inject"] == 2
    # The forced re-copy starts from the pristine edit, so the track is not injected twice
    assert (out / pair.name / "GH010234.mp4").read_bytes() == b"edited video-gpmd-track"


def test_process_pair_records_a_rejected_alignment(tmp_path, stub_externals, monkeypatch):
    monkeypatch.setattr(preproc, "align", lambda pair, **kw: preproc.Alignment(0.0, 0.2, False))
    pair = _pair(tmp_path)

    record = preproc.process_pair(pair, tmp_path / "curated", preproc.DEFAULT_ALIGN_MIN_R, force=False)

    assert record["aligned"] is False


def test_main_warns_about_every_unaligned_clip(tree, tmp_path, monkeypatch, caplog):
    # The summary's one job a human must not miss: which clips carry source-time telemetry
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        preproc,
        "process_pair",
        lambda pair, out, min_r, force: {
            "name": pair.name, "status": "processed",
            "aligned": False, "r": 0.2, "imu": True, "injected": False,
        },
    )
    with caplog.at_level("WARNING"):
        preproc.main(["--source-root", str(tree), "--output-root", str(tmp_path / "out")])

    assert caplog.text.count("alignment rejected") == 2


def test_main_survives_a_clip_that_raises(tree, tmp_path, monkeypatch, caplog):
    # A clip with no audio track makes ffmpeg exit 1 and decode_audio raise. Before this, the
    # exception left main after gigabytes had already been copied: no index, no summary.
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)

    def explode_on_one(pair, out, min_r, force):
        if "GH010228" in pair.name:
            raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="Output file does not contain any stream")
        return {"name": pair.name, "status": "processed", "aligned": True, "r": 0.99, "imu": True, "injected": True}

    monkeypatch.setattr(preproc, "process_pair", explode_on_one)
    out = tmp_path / "curated"
    with caplog.at_level("INFO"):
        assert preproc.main(["--source-root", str(tree), "--output-root", str(out)]) == 0

    # The run finished: the index was still written and the survivor still counted
    assert (out / "index.csv").is_file()
    assert "1 processed, 0 skipped, 1 failed" in caplog.text
    # And the casualty is named individually, not buried in a count
    assert "processing failed, nothing curated: 2026_07_15-Goprosplat-GH010228" in caplog.text


def test_main_dry_run_forwards_the_flag_to_the_push_script(tree, tmp_path, monkeypatch):
    # --dry-run --push used to return before the push block, so --push did nothing at all
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []
    monkeypatch.setattr(preproc, "push", lambda root, dry_run=False: calls.append(dry_run))

    preproc.main(["--source-root", str(tree), "--output-root", str(tmp_path / "out"), "--dry-run", "--push"])

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
        preproc.main(["--source-root", str(tree), "--output-root", str(tmp_path / "out"), "--only", "nope"])

    assert "no pairs matched" in caplog.text
