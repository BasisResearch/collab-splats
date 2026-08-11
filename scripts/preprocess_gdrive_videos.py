#!/usr/bin/env python3
"""Flatten the Google Drive capture tree and carry capture metadata into the curated videos.

DaVinci Resolve strips every timed data track and container tag on export: the camera
original carries ``tmcd + gpmd + fdsc``, GPS and the GoPro UserData block, while the export
carries ``tmcd`` alone. This script pairs each export against its original, copies it into
the flat curated layout, solves the trim offset from audio, and writes the metadata back —
as static tags plus a retimed ``gpmd`` track inside the mp4, and as a full-rate Parquet
sidecar next to it.

Source layout, where the parent folder name is not stable across dates (``GoproSplat``,
``Goprosplat``, ``splats``, ``Phone pics and splat videos``)::

    <source-root>/<YYYY-MM-DD>/<parent>/<stem>.mp4       <- Resolve export (the edit)
                                       /src/<stem>.MP4   <- camera original

A video is an *edit* if and only if ``<parent>/src/<stem>.*`` exists, matched on stem only,
case-insensitively, across any video extension. Only pairs are processed; an original with
no export is skipped and logged, and enters scope automatically once it is edited.

Output layout — one flat folder per pair::

    <output-root>/<YYYY_MM_DD>-<parent>-<stem>/
        <stem>.mp4                  edit + static tags + retimed gpmd track
        <stem>_metadata.json        static tags, provenance, alignment scores
        <stem>_telemetry.parquet    ~200 Hz IMU and GPS samples (only when IMU exists)
    <output-root>/index.csv         unique_id,gps — regenerated from the sidecars

Usage:
    python scripts/preprocess_gdrive_videos.py --dry-run
    python scripts/preprocess_gdrive_videos.py
    python scripts/preprocess_gdrive_videos.py --only GH010234
    python scripts/preprocess_gdrive_videos.py --index-only
    python scripts/preprocess_gdrive_videos.py --push
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

########
# Constants
########

_REPO_ROOT = Path(__file__).parent.parent
DEFAULT_SOURCE_ROOT = _REPO_ROOT.parent / "gdrive-src"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT.parent / "environments-curated"
PUSH_SCRIPT = _REPO_ROOT / "scripts" / "push_curated.sh"

_VIDEO_EXTS = {".mp4", ".mov", ".avi"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Audio is decoded mono at this rate purely to solve the trim offset; one sample is 125 us,
# far finer than a 59.94 fps frame, so the offset is frame-exact.
AUDIO_RATE = 8000

# The edit's audio is a literal subsegment of the source, so a true match correlates near
# 1.0 and a wrong lag lands near 1/sqrt(N). See the spec for why the gap is this wide.
DEFAULT_ALIGN_MIN_R = 0.95

INDEX_NAME = "index.csv"


########
# Naming
########


def sanitize(name):
    """Replace runs of non-alphanumeric characters with a single underscore."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def find_videos(folder):
    """Return videos sitting directly inside `folder`, sorted; never recurses."""
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in _VIDEO_EXTS)


def flat_dir_name(date, parent, video):
    """Build the flat folder name YYYY_MM_DD-PARENT-VIDEO for one video.

    The video stem is carried over verbatim so the folder is traceable back to the original
    filename (e.g. PXL_20260630_002106958.TS keeps its dot). Only the parent folder is
    sanitized. The stem is last, so any hyphens it contains stay unambiguous when splitting
    the name on the first two hyphens.
    """
    return f"{date.replace('-', '_')}-{sanitize(parent)}-{video.stem}"


########
# Pairing
########


class Pair(NamedTuple):
    """One Resolve export and the camera original it was cut from."""

    edit: Path
    source: Path
    name: str


def find_source(video):
    """Return the camera original for `video`, or None if it has no src/ counterpart.

    Matches on stem only: the real tree differs by case (GH010234.mp4 vs src/GH010234.MP4)
    and by extension (IMG_4085.mp4 vs src/IMG_4085.MOV), so neither can be compared directly.
    """
    src_dir = video.parent / "src"
    if not src_dir.is_dir():
        return None
    stem = video.stem.lower()
    for candidate in find_videos(src_dir):
        if candidate.stem.lower() == stem:
            return candidate
    return None


def plan_pairs(source_root):
    """Walk the capture tree and return every (edit, source) pair, sorted by flat name.

    A video is an edit iff <parent>/src/<stem>.* exists. Videos with no counterpart are
    camera originals that have not been edited yet; they are skipped and logged, and enter
    scope automatically once exported. Raises ValueError on a flat-name collision, which
    would otherwise silently overwrite one capture with another.
    """
    pairs = []
    skipped = 0
    for date_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if not _DATE_RE.fullmatch(date_dir.name):
            logger.info("skip %s: not a YYYY-MM-DD directory", date_dir.name)
            continue
        # Videos may sit directly in the date folder or one level down in a named parent.
        # The old plan_copies only looked one level down, so the former were invisible.
        folders = [date_dir] + sorted(p for p in date_dir.iterdir() if p.is_dir() and p.name != "src")
        for folder in folders:
            for video in find_videos(folder):
                source = find_source(video)
                if source is None:
                    logger.info("skip %s: no src/ counterpart (unedited original)", video.name)
                    skipped += 1
                    continue
                pairs.append(Pair(video, source, flat_dir_name(date_dir.name, folder.name, video)))

    seen = {}
    for pair in pairs:
        if pair.name in seen:
            raise ValueError(f"flat name collision {pair.name!r}: {seen[pair.name]} and {pair.edit}")
        seen[pair.name] = pair.edit

    logger.info("%d pairs, %d originals skipped", len(pairs), skipped)
    return sorted(pairs, key=lambda p: p.name)


########
# Copy and idempotency
########


def copy_video(video, dest_dir, force=False):
    """Copy `video` into `dest_dir` under its original filename; return bytes copied.

    Writes to a `.partial` sidecar and renames, so an interrupted run cannot leave a
    truncated file that looks complete. Returns 0 when an up-to-date copy already exists.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / video.name
    size = video.stat().st_size

    # Existing copy of the right size is treated as done unless forced
    if dest.exists() and not force and dest.stat().st_size == size:
        logger.info("exists  %s", dest.relative_to(dest_dir.parent))
        return 0

    logger.info("copying %s (%.1f GB)", dest.relative_to(dest_dir.parent), size / 1e9)
    partial = dest.with_suffix(dest.suffix + ".partial")
    shutil.copy2(video, partial)
    os.replace(partial, dest)
    return size


def metadata_path(dest_dir, video):
    """Return the JSON sidecar path for `video` inside `dest_dir`."""
    return dest_dir / f"{video.stem}_metadata.json"


def read_metadata(dest_dir, video):
    """Return the JSON sidecar as a dict, or None when it does not exist."""
    path = metadata_path(dest_dir, video)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_metadata(dest_dir, video, payload):
    """Write the JSON sidecar for `video` and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_path(dest_dir, video)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def source_fingerprint(path):
    """Return the size and mtime used to decide whether a curated copy is still current."""
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime": stat.st_mtime}


def needs_copy(pair, dest_dir, force=False):
    """Return True when the curated copy is missing or its recorded source has changed.

    The comparison is against the fingerprint stored in the JSON sidecar, never against the
    destination's own size. Injection adds a gpmd track and static tags, so the curated mp4
    ends up larger than the file it was copied from; a destination-size check would fail on
    every subsequent run, clobber the injected file, and re-upload ~7 GB each time.
    """
    if force:
        return True
    if not (dest_dir / pair.edit.name).is_file():
        return True
    recorded = (read_metadata(dest_dir, pair.edit) or {}).get("source")
    if not recorded:
        return True
    return recorded != source_fingerprint(pair.edit)


########
# GPS formatting
########


def format_dms(value, axis):
    """Format a signed decimal degree as degrees, minutes and 2-decimal seconds.

    The sign becomes the hemisphere letter, so the output never carries a leading minus:
    -71.0659 with axis="lon" gives 71 deg 3' 57.24" W. This matches the exiftool convention
    already in use when reading these files by hand.
    """
    hemisphere = ("N", "S") if axis == "lat" else ("E", "W")
    letter = hemisphere[0] if value >= 0 else hemisphere[1]
    # Round to hundredths of a second before splitting, so a value that rounds up to 60.00
    # carries into the minutes instead of producing a malformed "60.00" seconds field
    total_seconds = round(abs(value) * 3600, 2)
    degrees, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(degrees)} deg {int(minutes)}' {seconds:.2f}\" {letter}"


def format_gps(lat, lon):
    """Format a fix as a single human-readable string, or "" when there is no fix."""
    if lat is None or lon is None:
        return ""
    return f"{format_dms(lat, 'lat')}, {format_dms(lon, 'lon')}"


########
# CSV index
########


def index_rows(curated_root):
    """Scan the curated tree and return (unique_id, gps) rows sorted by unique_id.

    Derived entirely from the JSON sidecars, so the index can never drift from them and a
    partial run still produces a complete file: clips this run did not touch still have
    their sidecar on disk.
    """
    rows = []
    for sidecar in sorted(curated_root.glob("*/*_metadata.json")):
        payload = json.loads(sidecar.read_text())
        gps = payload.get("gps") or {}
        rows.append(
            (
                payload.get("unique_id", sidecar.parent.name),
                format_gps(gps.get("latitude"), gps.get("longitude")),
            )
        )
    return sorted(rows)


def write_index(curated_root):
    """Regenerate <curated-root>/index.csv from the sidecars and return its path."""
    # --index-only may run on a machine that has never produced this directory
    curated_root.mkdir(parents=True, exist_ok=True)
    rows = index_rows(curated_root)
    path = curated_root / INDEX_NAME
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unique_id", "gps"])
        writer.writerows(rows)
    logger.info("index: %d rows -> %s", len(rows), path)
    return path


########
# Alignment
########


class Alignment(NamedTuple):
    """Where the edit sits inside its source, and how well the audio matched."""

    offset_s: float
    r: float
    ok: bool


def solve_offset(edit, source, rate=AUDIO_RATE, min_r=DEFAULT_ALIGN_MIN_R):
    """Locate `edit` inside `source` by normalized cross-correlation.

    Resolve cuts and colour-corrects but never alters audio content, so the edit's audio is
    a literal subsegment of the source and a true match correlates near 1.0. A wrong lag
    over N samples lands near 1/sqrt(N) — about 0.0006 for a minute at 8 kHz — so the two
    cases are three orders of magnitude apart and the 0.95 gate is not delicate.

    The lag search is restricted to feasible positions, so an offset that would run the edit
    past the end of the source cannot be returned at all.
    """
    n, m = len(edit), len(source)
    if n == 0 or n > m:
        logger.warning("alignment impossible: edit has %d samples, source has %d", n, m)
        return Alignment(0.0, 0.0, False)

    # Remove the DC component from both so the correlation measures shape, not offset
    e = np.asarray(edit, dtype=np.float64) - np.mean(edit)
    s = np.asarray(source, dtype=np.float64) - np.mean(source)
    e_energy = float(np.sqrt(np.dot(e, e)))
    if e_energy == 0.0:
        logger.warning("alignment impossible: the edit's audio has no variance (silence)")
        return Alignment(0.0, 0.0, False)

    # Correlation at every feasible lag; "valid" gives exactly m - n + 1 positions
    numerator = signal.fftconvolve(s, e[::-1], mode="valid")
    # Energy of each length-n window of the source, via a cumulative sum
    cumulative = np.concatenate(([0.0], np.cumsum(s * s)))
    window_energy = np.sqrt(np.maximum(cumulative[n:] - cumulative[:-n], 0.0))
    denominator = window_energy * e_energy
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denominator > 0, numerator / denominator, 0.0)

    lag = int(np.argmax(r))
    peak = float(r[lag])
    return Alignment(lag / rate, peak, peak >= min_r)


def decode_audio(path, rate=AUDIO_RATE):
    """Decode `path` to a mono float32 array at `rate` Hz via ffmpeg."""
    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def align(pair, rate=AUDIO_RATE, min_r=DEFAULT_ALIGN_MIN_R):
    """Solve where the edit sits inside its camera original, from audio alone."""
    result = solve_offset(decode_audio(pair.edit, rate), decode_audio(pair.source, rate), rate, min_r)
    if result.ok:
        logger.info("%s: offset %.3f s (r=%.4f)", pair.name, result.offset_s, result.r)
    else:
        logger.warning("%s: alignment rejected (r=%.4f < %.2f)", pair.name, result.r, min_r)
    return result


########
# Metadata extraction
########


def _ungrouped(doc):
    """Strip exiftool's `-G3` document prefix, so Doc12:SampleTime becomes SampleTime."""
    return {key.split(":", 1)[-1]: value for key, value in doc.items()}


def exif_dump(path):
    """Return exiftool's full JSON dump for `path`, timed metadata included.

    -ee walks the embedded documents, which is what surfaces the per-chunk GPMF payloads;
    -n gives raw numbers rather than formatted strings; -G3 tags each value with the
    document it came from, which is how samples stay grouped by 1 Hz chunk.
    """
    command = ["exiftool", "-ee", "-api", "LargeFileSupport=1", "-json", "-n", "-G3", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def probe_duration(path):
    """Return the container duration of `path` in seconds."""
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ]
    return float(subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip())


def find_gpmd_index(path):
    """Return the stream index of the gpmd data track, or None when there is not one.

    GPMF rides a `bin_data` stream tagged `gpmd`; Sony's rtmd and the phone tracks are
    different tags, so matching on the tag is what keeps injection GoPro-only for now.
    """
    command = [
        "ffprobe", "-v", "error", "-select_streams", "d",
        "-show_entries", "stream=index,codec_tag_string", "-of", "json", str(path),
    ]
    streams = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    for stream in streams.get("streams", []):
        if stream.get("codec_tag_string") == "gpmd":
            return int(stream["index"])
    return None


def static_tags(dump):
    """Return the container-level tags that are true of the edit regardless of the trim."""
    wanted = (
        "Make", "Model", "SerialNumber", "FirmwareVersion", "CreateDate", "MediaCreateDate",
        "FieldOfView", "LensProjection", "ProjectionType", "ElectronicImageStabilization",
        "GPSCoordinates", "GPSAltitude",
    )
    tags = {}
    for doc in dump:
        flat = _ungrouped(doc)
        for key in wanted:
            if key in flat and key not in tags:
                tags[key] = flat[key]
    return tags


def first_fix(dump, start_s=0.0):
    """Return the first locked GPS fix at or after `start_s`, or None when there is none.

    A GoPro emits 0,0 before satellite lock, so those samples are discarded rather than
    trusted — 4 of the 33 originals never lock at all. Clips with no GPS track (Pixel,
    iPhone) fall back to the single container fix, which has no time and is reported at 0.
    """
    best = None
    for doc in dump:
        flat = _ungrouped(doc)
        lat, lon = flat.get("GPSLatitude"), flat.get("GPSLongitude")
        if lat is None or lon is None:
            continue
        if float(lat) == 0.0 and float(lon) == 0.0:
            continue
        when = float(flat.get("SampleTime", 0.0))
        if when < start_s:
            continue
        if best is None or when < best["source_time"]:
            best = {"latitude": float(lat), "longitude": float(lon), "source_time": when}
    return best


def has_gps_track(dump):
    """Return True when the dump carries timed GPS samples rather than one container fix."""
    for doc in dump:
        flat = _ungrouped(doc)
        if "SampleTime" in flat and flat.get("GPSLatitude") is not None and flat.get("GPSLongitude") is not None:
            return True
    return False


def first_chunk_at_or_after(dump, start_s):
    """Return the smallest GPMF SampleTime at or after `start_s`, or None when there is none."""
    later = []
    for doc in dump:
        when = _ungrouped(doc).get("SampleTime")
        if when is not None and float(when) >= start_s:
            later.append(float(when))
    return min(later) if later else None


def build_payload(pair, dump, alignment, duration_s, unique_id, has_imu, fingerprint):
    """Assemble the JSON sidecar for one pair.

    `gps` is the first locked fix inside the trimmed edit when alignment succeeded, and the
    first fix anywhere in the source when it did not — the edit window is unknown in that
    case, and `gps_source_anchored` records it so the value is never mistaken for exact. A
    successful alignment whose own window never locks (satellite fix acquired just after the
    cut, say) falls back to that same whole-source search rather than reporting no GPS at all.
    """
    fix = first_fix(dump, start_s=alignment.offset_s) if alignment.ok else None
    anchored = fix is None
    if anchored:
        fix = first_fix(dump, start_s=0.0)
    # A container-only fix (phone clips) is one untimed whole-file coordinate reported at 0,
    # so any non-zero offset skips it and falls through to the whole-source search. That is
    # not the same as losing a timed fix, so it must not be labelled source-anchored.
    anchored = anchored and fix is not None and has_gps_track(dump)

    # GPMF chunks are 1 Hz, so the injected track starts on the first chunk boundary at or
    # after the cut, up to ~1 s late. The residual is what a consumer needs to reconcile the
    # injected track against the Parquet sidecar, which stays on the true source time axis.
    first_chunk = first_chunk_at_or_after(dump, alignment.offset_s)
    return {
        "unique_id": unique_id,
        "edit": str(pair.edit),
        "source": fingerprint,
        "source_path": str(pair.source),
        "duration_s": duration_s,
        "has_imu": has_imu,
        "gps": fix,
        "gps_source_anchored": anchored,
        "gpmd_first_chunk_s": first_chunk,
        "gpmd_residual_s": None if first_chunk is None else first_chunk - alignment.offset_s,
        "alignment": {"offset_s": alignment.offset_s, "r": alignment.r, "ok": alignment.ok},
        "static_tags": static_tags(dump),
        # The edits are cuts plus colour correction with no geometric transform, so lens
        # geometry still describes the exported pixels. If a clip is ever reframed or
        # stabilized, this flag is what has to flip.
        "intrinsics": {"valid_for_edit": True, "reason": "cuts and colour correction only, no reframe"},
    }


########
# Telemetry
########

# Column prefix -> (exiftool tag, component count). These are the wide tags: one chunk holds
# a flat run of N-tuples that has to be sliced back apart.
#
# VERIFY AGAINST REAL FOOTAGE: the exiftool tag names below are the documented GPMF stream
# names, but exiftool renames some of them per firmware. Run
#   exiftool -ee -api LargeFileSupport=1 -json -n -G3 <original> | python -c \
#     "import json,sys; print(sorted({k.split(':',1)[-1] for d in json.load(sys.stdin) for k in d}))"
# on one GoPro original and correct any mismatch before relying on the output. A wrong name
# yields an empty column, not an exception, so this fails quietly if left unchecked.
_GPMF_STREAMS = {
    "accl": ("Accelerometer", 3),
    "gyro": ("Gyroscope", 3),
    "grav": ("GravityVector", 3),
    "cori": ("CameraOrientation", 4),
    "iori": ("ImageOrientation", 4),
}
_AXES = {3: ("x", "y", "z"), 4: ("w", "x", "y", "z")}

# GPS is not a wide tag. exiftool splits the GPMF GPS stream into parallel single-value tags,
# one document per fix, so it is read by zipping them rather than by slicing one run. The tag
# formerly used here, a 5-wide "GPSTrack", was wrong twice over: exiftool's GPSTrack is the
# scalar heading from the EXIF GPS group, not the GPMF stream, so every chunk logged
# "skipping ragged GPSTrack chunk: 1 values for 5 components" and gps_* was always empty.
# Any of these may be absent on a given firmware; a missing one leaves its column null.
_GPMF_GPS_TAGS = ("GPSLatitude", "GPSLongitude", "GPSAltitude", "GPSSpeed", "GPSSpeed3D")
_GPMF_GPS_AXES = ("lat", "lon", "alt", "speed2d", "speed3d")


def expand_gpmf(dump, key, components):
    """Expand 1 Hz GPMF chunks into per-sample (times, values) on the source time axis.

    exiftool returns one value-set per chunk with every sample concatenated — a single
    Accelerometer chunk is a flat run of triplets — plus SampleTime and SampleDuration.
    Samples are spread evenly across the chunk, which is the best placement available:
    GPMF carries no per-sample timestamp, only a per-chunk one.
    """
    times, values = [], []
    for doc in dump:
        flat = _ungrouped(doc)
        raw = flat.get(key)
        if raw is None:
            continue
        numbers = [float(x) for x in (raw if isinstance(raw, list) else str(raw).split())]
        if not numbers or len(numbers) % components:
            logger.warning("skipping ragged %s chunk: %d values for %d components", key, len(numbers), components)
            continue
        count = len(numbers) // components
        start = float(flat.get("SampleTime", 0.0))
        step = float(flat.get("SampleDuration", 1.0)) / count
        for i in range(count):
            times.append(start + i * step)
            values.append(tuple(numbers[i * components:(i + 1) * components]))
    return times, values


def expand_gpmf_parallel(dump, keys):
    """Zip parallel single-value GPMF tags into one (times, values) row per timed document.

    exiftool emits the GPS stream as one document per fix carrying GPSLatitude, GPSLongitude
    and friends side by side, rather than as one wide tag. A key that is absent yields None
    for that component, so a firmware that drops GPSSpeed3D still produces the other columns.
    Documents with no SampleTime are skipped: the container-level document carries GPSAltitude
    and a single whole-file coordinate, and that is not a timed sample.
    """
    times, values = [], []
    for doc in dump:
        flat = _ungrouped(doc)
        if "SampleTime" not in flat:
            continue
        row = tuple(None if flat.get(key) is None else float(flat[key]) for key in keys)
        if all(value is None for value in row):
            continue
        times.append(float(flat["SampleTime"]))
        values.append(row)
    return times, values


def telemetry_table(dump, alignment, duration_s):
    """Build the per-sample telemetry table, or None when no configured stream produced data.

    The streams do not share a time axis. Each GPMF chunk carries its own sample count, so the
    per-sample step `SampleDuration / count` differs from stream to stream and the resulting
    axes are very nearly disjoint. This table is therefore the *union* of every stream's sample
    times, and each column is sparse: a typical row carries one stream's values and nulls in
    every other column, so the row count runs well above any single stream's rate. Nothing is
    resampled — a null means "this stream has no sample at this instant", not missing data. The
    fill ratio of each column is logged so that sparsity is observable rather than a surprise.

    `edit_time` and `in_edit` are null when alignment failed: the cut window is unknown, so any
    value would be a guess.
    """
    streams = {}
    for prefix, (key, components) in _GPMF_STREAMS.items():
        times, values = expand_gpmf(dump, key, components)
        if times:
            streams[prefix] = (times, values, _AXES[components])
    gps_times, gps_values = expand_gpmf_parallel(dump, _GPMF_GPS_TAGS)
    if gps_times:
        streams["gps"] = (gps_times, gps_values, _GPMF_GPS_AXES)
    if not streams:
        return None

    axis = sorted({t for times, _, _ in streams.values() for t in times})
    columns = {"source_time": axis}
    for prefix, (times, values, axes) in streams.items():
        lookup = dict(zip(times, values))
        for i, name in enumerate(axes):
            columns[f"{prefix}_{name}"] = [lookup[t][i] if t in lookup else None for t in axis]

    if alignment.ok:
        columns["edit_time"] = [t - alignment.offset_s for t in axis]
        end = alignment.offset_s + duration_s
        columns["in_edit"] = [alignment.offset_s <= t <= end for t in axis]
    else:
        columns["edit_time"] = [None] * len(axis)
        columns["in_edit"] = [None] * len(axis)

    table = pa.table(columns)
    # Report the real sparsity of the union axis, so a reader never has to guess whether a
    # half-null column means a wrong tag name or simply a stream on its own clock
    fills = [
        f"{name}={1.0 - table.column(name).null_count / table.num_rows:.0%}"
        for name in table.column_names
        if name != "source_time"
    ]
    logger.info("telemetry: %d rows on the union time axis, fill %s", table.num_rows, " ".join(fills))
    return table


def write_telemetry(table, dest_dir, video):
    """Write the telemetry table beside the curated video and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{video.stem}_telemetry.parquet"
    pq.write_table(table, path, compression="zstd")
    logger.info("telemetry: %d rows -> %s", table.num_rows, path.name)
    return path


########
# Injection
########

# Written into the curated file so a re-run can tell an injected mp4 from a fresh copy
PROVENANCE_TAG = "preprocess_gdrive_videos"

# The subset of static_tags exiftool will actually write. LensProjection and
# ElectronicImageStabilization are read-only derived tags — exiftool answers "Sorry, <tag> is
# not writable" and, when other tags in the same call do land, still exits 0. Determined
# empirically with `exiftool -overwrite_original -n -<tag>=<value>` per tag on a synthesized
# mp4; the two rejected ones stay in static_tags so the JSON sidecar keeps carrying them.
_WRITABLE_TAGS = frozenset(
    {
        "Make", "Model", "SerialNumber", "FirmwareVersion", "CreateDate", "MediaCreateDate",
        "FieldOfView", "ProjectionType", "GPSCoordinates", "GPSAltitude",
    }
)


def gpmd_command(curated, source, offset_s, duration_s, gpmd_index, out):
    """Build the ffmpeg call that grafts the trimmed gpmd track onto the curated video.

    -ss and -t precede the second -i so they trim that input rather than the first. -c copy
    leaves the edit's pixels untouched, and -copy_unknown is what allows the bin_data gpmd
    stream through at all — without it ffmpeg silently drops unrecognized track types.
    """
    return [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(curated),
        "-ss", f"{offset_s}", "-t", f"{duration_s}", "-i", str(source),
        "-map", "0", "-map", f"1:{gpmd_index}",
        "-c", "copy", "-copy_unknown",
        str(out),
    ]


def tag_command(curated, tags):
    """Build the exiftool call that writes static tags into the finished container."""
    # -n is not optional. exif_dump reads with -n, so GPSCoordinates arrives already raw
    # ("42.3532 -71.0659 5.2"); writing it back without -n runs PrintConvInv over a raw value,
    # which warns, writes nothing for that tag, and still exits 0 whenever any other tag lands.
    command = ["exiftool", "-overwrite_original", "-api", "QuickTimeUTC", "-n"]
    for key, value in tags.items():
        if value in (None, "") or key not in _WRITABLE_TAGS:
            continue
        command.append(f"-{key}={value}")
    command.append(f"-Software={PROVENANCE_TAG}")
    command.append(str(curated))
    return command


def _last_stderr_line(text):
    """Return the last non-empty stderr line, so a failure logs as one readable line."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "(no stderr)"


def inject(curated, source, alignment, duration_s, tags):
    """Write metadata back into the curated video; return True when a gpmd track landed.

    Order matters. ffmpeg runs first because it rewrites the container; exiftool runs second
    to write tags into the final one. Reversed, ffmpeg drops the tags. The remux goes to a
    temporary file that is only moved into place on success, so a failure leaves the curated
    video exactly as it was. A failure in either external tool is logged and the run continues;
    the return value reports only whether a gpmd track landed.
    """
    injected = False
    gpmd_index = find_gpmd_index(source) if alignment.ok else None
    if not alignment.ok:
        logger.warning("%s: no gpmd injected, alignment was rejected", curated.name)
    elif gpmd_index is None:
        logger.info("%s: no gpmd track in the source, writing static tags only", curated.name)
    else:
        # ".inject" goes before the extension, not after: ffmpeg infers the output muxer from
        # the final suffix, and "GH010234.mp4.inject" makes it exit 1 with "Unable to find a
        # suitable output format" before it reads a single frame.
        temp = curated.with_suffix(".inject" + curated.suffix)
        command = gpmd_command(curated, source, alignment.offset_s, duration_s, gpmd_index, temp)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            os.replace(temp, curated)
            injected = True
        else:
            temp.unlink(missing_ok=True)
            logger.error("gpmd injection failed for %s: %s", curated.name, _last_stderr_line(result.stderr))

    # Logged rather than raised: one clip with unwritable tags must not abort the run.
    # exiftool's exit code cannot carry this signal — a tag that fails to convert or is not
    # writable is reported only as a stderr warning, and the process still exits 0 as long as
    # some other tag in the same call landed. So the stderr text is the check, not the code.
    tagged = subprocess.run(tag_command(curated, tags), capture_output=True, text=True, check=False)
    for line in tagged.stderr.splitlines():
        if "Warning:" in line or "Error" in line:
            logger.warning("static tags for %s: %s", curated.name, line.strip())
    if tagged.returncode != 0:
        logger.error("static tags failed for %s: %s", curated.name, _last_stderr_line(tagged.stderr))
    return injected


########
# Pipeline
########


def require_binaries():
    """Exit before any work when a required external tool is missing."""
    missing = [name for name in ("ffmpeg", "ffprobe", "exiftool") if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"missing required tools: {', '.join(missing)}")


def push(output_root, dry_run=False):
    """Run the rclone push script over `output_root`, forwarding --dry-run."""
    command = [str(PUSH_SCRIPT), "--source", str(output_root)]
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


def process_pair(pair, output_root, min_r, force):
    """Run copy, align, extract and inject for one pair; return a summary record."""
    dest_dir = output_root / pair.name
    curated = dest_dir / pair.edit.name

    if needs_copy(pair, dest_dir, force=force):
        copy_video(pair.edit, dest_dir, force=True)
    else:
        # needs_copy already returns True whenever force is set, so reaching here means
        # the copy is current and not forced: there is nothing left to do
        logger.info("%s: up to date", pair.name)
        return {"name": pair.name, "status": "skipped"}

    alignment = align(pair, min_r=min_r)
    dump = exif_dump(pair.source)
    duration_s = probe_duration(curated)

    table = telemetry_table(dump, alignment, duration_s)
    if table is not None:
        write_telemetry(table, dest_dir, pair.edit)

    payload = build_payload(
        pair, dump, alignment, duration_s, pair.name, table is not None, source_fingerprint(pair.edit)
    )
    injected = inject(curated, pair.source, alignment, duration_s, payload["static_tags"])
    payload["gpmd_injected"] = injected
    write_metadata(dest_dir, pair.edit, payload)

    return {
        "name": pair.name,
        "status": "processed",
        "aligned": alignment.ok,
        "r": alignment.r,
        "imu": table is not None,
        "injected": injected,
    }


########
# Entry point
########


def main(argv=None):
    """Parse args and run the curation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="nested capture tree to read")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="flat curated tree to write")
    parser.add_argument("--only", help="process only pairs whose flat name contains this substring")
    parser.add_argument("--dry-run", action="store_true", help="log the plan and total size, change nothing")
    parser.add_argument("--force", action="store_true", help="re-copy and re-inject even when up to date")
    parser.add_argument("--index-only", action="store_true", help="rebuild index.csv from the sidecars and stop")
    parser.add_argument("--push", action="store_true", help="run scripts/push_curated.sh after processing")
    parser.add_argument("--align-min-r", type=float, default=DEFAULT_ALIGN_MIN_R, help="minimum correlation to accept")
    args = parser.parse_args(argv)

    # --index-only reads only the JSON sidecars, so it must not demand ffmpeg or exiftool;
    # the whole point is that it works on a machine with no Drive mount and no media tools
    if args.index_only:
        write_index(args.output_root)
        return 0

    require_binaries()

    if not args.source_root.is_dir():
        parser.error(f"source root does not exist: {args.source_root}")

    pairs = plan_pairs(args.source_root)
    if args.only:
        pairs = [p for p in pairs if args.only in p.name]
    if not pairs:
        if args.only:
            logger.warning("no pairs matched --only %r under %s", args.only, args.source_root)
        else:
            logger.warning("no pairs found under %s", args.source_root)
        return 0

    if args.dry_run:
        total = sum(p.edit.stat().st_size for p in pairs)
        for pair in pairs:
            logger.info("%s/%s  <- %s", pair.name, pair.edit.name, pair.source)
        logger.info("dry run: %d pairs, %.1f GB", len(pairs), total / 1e9)
        # A dry run that was also asked to push forwards the flag rather than dropping it
        if args.push:
            push(args.output_root, dry_run=True)
        return 0

    # A single unreadable clip — no audio track, a truncated download — must not throw away a
    # multi-hour run that has already copied gigabytes. Record the failure and keep going, so
    # write_index and the summary still run.
    records = []
    for pair in pairs:
        try:
            records.append(process_pair(pair, args.output_root, args.align_min_r, args.force))
        except Exception:
            logger.exception("%s: failed, skipping", pair.name)
            records.append({"name": pair.name, "status": "failed"})
    write_index(args.output_root)

    processed = [r for r in records if r["status"] == "processed"]
    failed = [r["name"] for r in records if r["status"] == "failed"]
    unaligned = [r["name"] for r in processed if not r["aligned"]]
    logger.info(
        "done: %d processed, %d skipped, %d failed, %d with IMU, %d injected",
        len(processed),
        sum(1 for r in records if r["status"] == "skipped"),
        len(failed),
        sum(1 for r in processed if r["imu"]),
        sum(1 for r in processed if r["injected"]),
    )
    # Surfaced loudly: these clips carry source-time telemetry and no injected track
    for name in unaligned:
        logger.warning("alignment rejected, telemetry left in source time: %s", name)
    # Likewise: nothing was written for these at all, so they need a human
    for name in failed:
        logger.warning("processing failed, nothing curated: %s", name)

    if args.push:
        push(args.output_root, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
