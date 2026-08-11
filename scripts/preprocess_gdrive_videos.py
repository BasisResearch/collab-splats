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
    return {
        "unique_id": unique_id,
        "edit": str(pair.edit),
        "source": fingerprint,
        "source_path": str(pair.source),
        "duration_s": duration_s,
        "has_imu": has_imu,
        "gps": fix,
        "gps_source_anchored": anchored and fix is not None,
        "alignment": {"offset_s": alignment.offset_s, "r": alignment.r, "ok": alignment.ok},
        "static_tags": static_tags(dump),
        # The edits are cuts plus colour correction with no geometric transform, so lens
        # geometry still describes the exported pixels. If a clip is ever reframed or
        # stabilized, this flag is what has to flip.
        "intrinsics": {"valid_for_edit": True, "reason": "cuts and colour correction only, no reframe"},
    }
