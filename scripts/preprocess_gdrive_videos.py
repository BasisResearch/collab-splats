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
