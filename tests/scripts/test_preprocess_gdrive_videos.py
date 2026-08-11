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
