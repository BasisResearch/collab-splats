#!/usr/bin/env python3
"""
Flatten the Google Drive capture tree into curated videos that keep their capture metadata.

- edit: a video with a `<parent>/src/<stem>.*` original, matched on stem, case-insensitive
- original: any other video outside `src/`; it is its own metadata source
- edits are aligned to their original by audio, then get static tags and a retimed `gpmd` track
- every video gets a JSON sidecar and, when it has IMU, a Parquet telemetry sidecar
- `src/` videos with nothing above them are orphans: logged, never curated
- `--prune-src` reports (`--apply` deletes) `src/` copies identical to the video above them

Layout:
    <output-root>/<component>-...-<stem>/<stem>.mp4
    <output-root>/<component>-...-<stem>/<stem>_metadata.json
    <output-root>/<component>-...-<stem>/<stem>_telemetry.parquet
    <output-root>/index.csv

Usage:
    python scripts/preprocess_gdrive_videos.py --dry-run
    python scripts/preprocess_gdrive_videos.py --only <stem>
    python scripts/preprocess_gdrive_videos.py --prune-src [--apply]
    python scripts/preprocess_gdrive_videos.py --index-only
    python scripts/preprocess_gdrive_videos.py --push
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal

logger = logging.getLogger(__name__)

########
# Constants
########

_REPO_ROOT = Path(__file__).parent.parent
DEFAULT_SOURCE_ROOT = _REPO_ROOT.parent / "gdrive-src"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT.parent / "environments-curated"
PUSH_SCRIPT = _REPO_ROOT / "scripts" / "push_curated.sh"

_VIDEO_EXTS = {".mp4", ".mov", ".avi"}

# Camera-original directory, case-sensitive so the walk and find_source agree
SRC_DIR_NAME = "src"

# exiftool -G3 groups: "Doc7" is one 1 Hz GPMF chunk, "Doc7-3" a GPS fix inside it
_DOC_GROUP_RE = re.compile(r"Doc(\d+)(?:-(\d+))?")

INDEX_NAME = "index.csv"


########
# Naming
########


def sanitize(name: str) -> str:
    """
    Collapse runs of non-alphanumeric characters to one underscore.

    Args:
        name: one path component.

    Returns:
        The component with leading and trailing underscores stripped.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def find_videos(folder: Path) -> list[Path]:
    """
    Videos directly inside a folder, sorted; never recurses.

    - dotfiles are skipped: macOS AppleDouble "._X.MP4" sidecars and .DS_Store

    Args:
        folder: directory to list.

    Returns:
        Video paths with a known extension.
    """
    return sorted(
        f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in _VIDEO_EXTS
    )


def flat_dir_name(video: Path, source_root: Path) -> str:
    """
    Flat folder name: sanitized directory components joined by hyphens, then the stem.

    - the stem is kept verbatim, so the folder traces back to the original filename
    - the stem goes last, so hyphens inside it stay unambiguous

    Args:
        video: path of the video below `source_root`.
        source_root: root of the capture tree.

    Returns:
        e.g. "2026_07_22-splats-<stem>".
    """
    parts = [sanitize(part) for part in video.parent.relative_to(source_root).parts]
    return "-".join(parts + [video.stem])


########
# Pairing
########


def find_source(video: Path) -> Path | None:
    """
    Camera original for a video, matched on stem only.

    - case and extension both differ in the real tree (".mp4" vs "src/*.MP4", ".mp4" vs ".MOV")

    Args:
        video: the candidate edit.

    Returns:
        The `src/` counterpart, or None when there is none.
    """
    src_dir = video.parent / SRC_DIR_NAME
    if not src_dir.is_dir():
        return None
    stem = video.stem.lower()
    for candidate in find_videos(src_dir):
        if candidate.stem.lower() == stem:
            return candidate
    return None


def find_orphans(folder: Path) -> list[Path]:
    """
    Videos in a folder's `src/` with no counterpart in the folder itself.

    Args:
        folder: directory that may hold a `src/`.

    Returns:
        Orphaned `src/` videos.
    """
    src_dir = folder / SRC_DIR_NAME
    if not src_dir.is_dir():
        return []
    stems = {video.stem.lower() for video in find_videos(folder)}
    return [candidate for candidate in find_videos(src_dir) if candidate.stem.lower() not in stems]


def walk_folders(root: Path) -> Iterator[Path]:
    """
    Every directory that may hold videos, depth-first, root first.

    - `src/` is pruned: its videos are reached through the folder above
    - dotted directories are pruned

    Args:
        root: capture tree root.

    Yields:
        Directory paths.
    """
    yield root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if child.name == SRC_DIR_NAME or child.name.startswith("."):
            continue
        yield from walk_folders(child)


def plan_videos(source_root: Path) -> list[tuple[Path, Path | None, str]]:
    """
    Every curatable video in the tree with its source and flat name.

    - an edit carries its `src/` original; an original carries None
    - orphans in `src/` are logged and skipped
    - the walk is depth-agnostic: any nesting is curated with no code change

    Args:
        source_root: capture tree root.

    Returns:
        (video, source_or_None, flat_name) tuples, sorted by flat name.

    Raises:
        ValueError: two videos map to the same flat name.
    """
    entries = []
    orphans = 0
    for folder in walk_folders(source_root):
        for video in find_videos(folder):
            entries.append((video, find_source(video), flat_dir_name(video, source_root)))
        for orphan in find_orphans(folder):
            logger.warning("orphan %s: left in src/ with no video above it, not curated", orphan)
            orphans += 1

    seen = {}
    for video, _, name in entries:
        if name in seen:
            raise ValueError(f"flat name collision {name!r}: {seen[name]} and {video}")
        seen[name] = video

    edits = sum(1 for _, source, _ in entries if source is not None)
    logger.info(
        "%d videos: %d edits, %d originals; %d orphans skipped", len(entries), edits, len(entries) - edits, orphans
    )
    return sorted(entries, key=lambda entry: entry[2])


########
# Copy and idempotency
########


def copy_video(video: Path, dest_dir: Path, force: bool = False) -> int:
    """
    Copy a video into its curated folder under its original filename.

    - writes a `.partial` file then renames, so an interrupted copy never looks complete

    Args:
        video: file to copy.
        dest_dir: curated folder.
        force: copy even when a same-size copy exists.

    Returns:
        Bytes copied; 0 when an up-to-date copy already existed.
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


def metadata_path(dest_dir: Path, video: Path) -> Path:
    """
    JSON sidecar path for a video.

    Args:
        dest_dir: curated folder.
        video: the curated video.

    Returns:
        `<dest_dir>/<stem>_metadata.json`.
    """
    return dest_dir / f"{video.stem}_metadata.json"


def read_metadata(dest_dir: Path, video: Path) -> dict | None:
    """
    Load a video's JSON sidecar.

    Args:
        dest_dir: curated folder.
        video: the curated video.

    Returns:
        The sidecar dict, or None when it does not exist.
    """
    path = metadata_path(dest_dir, video)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_metadata(dest_dir: Path, video: Path, payload: dict) -> Path:
    """
    Write a video's JSON sidecar.

    Args:
        dest_dir: curated folder.
        video: the curated video.
        payload: sidecar contents.

    Returns:
        The sidecar path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_path(dest_dir, video)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def source_fingerprint(path: Path) -> dict:
    """
    Size and mtime that decide whether a curated copy is still current.

    Args:
        path: the source video.

    Returns:
        {"size_bytes", "mtime"}.
    """
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime": stat.st_mtime}


def needs_copy(video: Path, dest_dir: Path, force: bool = False) -> bool:
    """
    Whether the curated copy is missing or its recorded source changed.

    - compares the fingerprint stored in the sidecar, never the destination size
    - injection grows the curated file, so a size check would re-copy every run

    Args:
        video: the source video.
        dest_dir: curated folder.
        force: always copy.

    Returns:
        True when the video must be copied.
    """
    if force:
        return True
    if not (dest_dir / video.name).is_file():
        return True
    recorded = read_metadata(dest_dir, video)
    if recorded is None:
        return True
    return recorded.get("source") != source_fingerprint(video)


def needs_refresh(video: Path, source: Path | None, dest_dir: Path) -> bool:
    """
    Whether a current copy's sidecar no longer describes it.

    - catches an edit reclassified as an original after its `src/` copy was pruned
    - metadata is rewritten in place: no re-copy, no re-injection

    Args:
        video: the source video.
        source: its camera original, or None for an original.
        dest_dir: curated folder.

    Returns:
        True when the sidecar must be rewritten.
    """
    recorded = read_metadata(dest_dir, video)
    if recorded is None:
        return True
    if recorded.get("edited") != (source is not None):
        return True
    recorded_source = recorded.get("source_path")
    return recorded_source is None or not Path(recorded_source).exists()


########
# GPS formatting
########


def format_dms(value: float, axis: str) -> str:
    """
    Signed decimal degrees as degrees, minutes and 2-decimal seconds plus hemisphere.

    - the sign becomes the hemisphere letter, matching exiftool's output

    Args:
        value: decimal degrees.
        axis: "lat" or "lon".

    Returns:
        e.g. `71 deg 3' 57.24" W`.
    """
    hemisphere = ("N", "S") if axis == "lat" else ("E", "W")
    letter = hemisphere[0] if value >= 0 else hemisphere[1]

    # Round before splitting so 59.995 s carries into the minutes
    total_seconds = round(abs(value) * 3600, 2)
    degrees, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(degrees)} deg {int(minutes)}' {seconds:.2f}\" {letter}"


def format_gps(lat: float | None, lon: float | None) -> str:
    """
    One fix as a human-readable string.

    Args:
        lat: latitude in decimal degrees, or None.
        lon: longitude in decimal degrees, or None.

    Returns:
        "<lat dms>, <lon dms>", or "" without a fix.
    """
    if lat is None or lon is None:
        return ""
    return f"{format_dms(lat, 'lat')}, {format_dms(lon, 'lon')}"


########
# CSV index
########


def index_rows(curated_root: Path) -> list[tuple[str, str]]:
    """
    Index rows read from every JSON sidecar in the curated tree.

    - derived only from sidecars, so a partial run still yields a complete index

    Args:
        curated_root: flat curated tree.

    Returns:
        (unique_id, gps) rows sorted by unique_id.
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


def write_index(curated_root: Path) -> Path:
    """
    Regenerate `index.csv` from the sidecars.

    Args:
        curated_root: flat curated tree.

    Returns:
        The index path.
    """
    # --index-only may run where this directory was never produced
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


def solve_offset(
    edit: np.ndarray, source: np.ndarray, rate: int = 8000, min_r: float = 0.95, tolerance_s: float = 1.0
) -> dict:
    """
    Locate the edit's audio inside the source's by normalized cross-correlation.

    - the edit's audio is a literal subsegment, so a true match scores near 1.0
    - the edit may overrun the source end by tolerance_s (AAC priming)
    - an empty, silent or too-long edit is logged and returned with ok=False

    Args:
        edit: edit audio samples.
        source: source audio samples.
        rate: sample rate of both, Hz.
        min_r: correlation needed for ok=True; a true match scores near 1.0, a wrong lag near 1/sqrt(N).
        tolerance_s: slack past the source end, seconds.

    Returns:
        {"offset_s", "r", "ok"}, the shape the sidecar stores.
    """
    # Zero padding keeps an overrunning lag reachable; it scores 0 so cannot win
    source = np.concatenate([np.asarray(source, dtype=np.float64), np.zeros(round(tolerance_s * rate))])
    n, m = len(edit), len(source)
    if n == 0 or n > m:
        logger.warning("alignment impossible: edit has %d samples, source has %d", n, m)
        return {"offset_s": 0.0, "r": 0.0, "ok": False}

    # Remove the DC component from both so the correlation measures shape, not offset
    e = np.asarray(edit, dtype=np.float64) - np.mean(edit)
    s = np.asarray(source, dtype=np.float64) - np.mean(source)
    e_energy = float(np.sqrt(np.dot(e, e)))
    if e_energy == 0.0:
        logger.warning("alignment impossible: the edit's audio has no variance (silence)")
        return {"offset_s": 0.0, "r": 0.0, "ok": False}

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
    return {"offset_s": lag / rate, "r": peak, "ok": peak >= min_r}


def decode_audio(path: Path, rate: int = 8000) -> np.ndarray:
    """
    Decode a file's audio to mono float32 via ffmpeg.

    Args:
        path: media file.
        rate: output sample rate, Hz.

    Returns:
        (N,) float32 samples.
    """
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def align(video: Path, source: Path, name: str, rate: int = 8000, min_r: float = 0.95) -> dict:
    """
    Where an edit sits inside its camera original, from audio alone.

    Args:
        video: the edit.
        source: its camera original.
        name: flat name, for logging.
        rate: audio decode rate, Hz.
        min_r: correlation needed to accept.

    Returns:
        {"offset_s", "r", "ok"}.
    """
    result = solve_offset(decode_audio(video, rate), decode_audio(source, rate), rate, min_r)
    if result["ok"]:
        logger.info("%s: offset %.3f s (r=%.4f)", name, result["offset_s"], result["r"])
    else:
        logger.warning("%s: alignment rejected (r=%.4f < %.2f)", name, result["r"], min_r)
    return result


########
# Metadata extraction
########


def iter_documents(dump: list[dict]) -> tuple[dict, list[tuple[int, dict, list[dict]]]]:
    """
    Regroup an exiftool -G3 dump by embedded document.

    - exiftool returns one object; the key prefix names the document
    - `Main:X` is the container, `Doc7:X` a 1 Hz GPMF chunk, `Doc7-3:X` a GPS fix inside it
    - chunks come back in numeric order (Doc2 before Doc10), subs in theirs
    - unprefixed keys such as SourceFile join the container tags

    Args:
        dump: exiftool JSON output.

    Returns:
        (container tags, [(chunk number, chunk tags, [sub tags, ...]), ...]).
    """
    main, chunks, subs = {}, {}, {}
    for doc in dump:
        for key, value in doc.items():
            group, _, tag = key.rpartition(":")
            match = _DOC_GROUP_RE.fullmatch(group)
            if match is None:
                main[tag] = value
            elif match[2] is None:
                chunks.setdefault(int(match[1]), {})[tag] = value
            else:
                subs.setdefault((int(match[1]), int(match[2])), {})[tag] = value

    # A sub-group whose parent chunk carries no tags of its own still gets a chunk to hang off
    nested = {number: [] for number in chunks}
    for parent, index in sorted(subs):
        nested.setdefault(parent, []).append(subs[(parent, index)])
    return main, [(number, chunks.get(number, {}), nested[number]) for number in sorted(nested)]


def gps_fixes(chunk: tuple[int, dict, list[dict]]) -> list[tuple[float, dict]]:
    """
    One chunk's GPS fixes, spread evenly across the chunk's window.

    - the chunk carries one fix of its own plus one sub-group per further fix that second
    - GPMF timestamps only the chunk, so per-fix times are interpolated

    Args:
        chunk: (number, tags, subs) from iter_documents.

    Returns:
        (source_time, tags) per fix, earliest first.
    """
    _, tags, subs = chunk
    fixes = [tags] if "GPSLatitude" in tags or "GPSLongitude" in tags else []
    fixes += subs
    if not fixes:
        return []
    start = float(tags["SampleTime"])
    step = float(tags["SampleDuration"]) / len(fixes)
    return [(start + i * step, fix) for i, fix in enumerate(fixes)]


def exif_dump(path: Path) -> list[dict]:
    """
    exiftool's full JSON dump of a file, timed metadata included.

    - `-ee` walks embedded documents; `-n` gives raw numbers; `-G3` keeps document groups
    - `-b` is required: without it every wide GPMF tag is a "(Binary data …)" placeholder

    Args:
        path: media file.

    Returns:
        exiftool's JSON list.
    """
    command = ["exiftool", "-ee", "-api", "LargeFileSupport=1", "-json", "-n", "-G3", "-b", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def probe_duration(path: Path) -> float:
    """
    Container duration via ffprobe.

    Args:
        path: media file.

    Returns:
        Duration in seconds.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    return float(subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip())


def find_gpmd_index(path: Path) -> int | None:
    """
    Stream index of the GoPro `gpmd` data track.

    - matched on the codec tag, which keeps injection GoPro-only

    Args:
        path: media file.

    Returns:
        The stream index, or None when there is no gpmd track.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "d",
        "-show_entries",
        "stream=index,codec_tag_string",
        "-of",
        "json",
        str(path),
    ]
    streams = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    for stream in streams.get("streams", []):
        if stream.get("codec_tag_string") == "gpmd":
            return int(stream["index"])
    return None


def static_tags(dump: list[dict]) -> dict:
    """
    Container-level tags that hold for the edit regardless of the trim.

    - read from the Main group only; GPMF chunks reuse names such as GPSAltitude and Model

    Args:
        dump: exiftool JSON output.

    Returns:
        Tag name to value, for the tags present.
    """
    wanted = (
        "Make",
        "Model",
        "SerialNumber",
        "FirmwareVersion",
        "CreateDate",
        "MediaCreateDate",
        "FieldOfView",
        "LensProjection",
        "ProjectionType",
        "ElectronicImageStabilization",
        "GPSCoordinates",
        "GPSAltitude",
    )
    main, _ = iter_documents(dump)
    return {key: main[key] for key in wanted if key in main}


def first_fix(dump: list[dict], start_s: float = 0.0) -> dict | None:
    """
    First locked GPS fix at or after a source time.

    - 0,0 samples (before satellite lock) are discarded
    - clips with no GPS track use the untimed container fix, reported at 0

    Args:
        dump: exiftool JSON output.
        start_s: earliest source time to accept, seconds.

    Returns:
        {"latitude", "longitude", "source_time"}, or None.
    """
    main, chunks = iter_documents(dump)
    timed = [chunk for chunk in chunks if "SampleTime" in chunk[1]]
    for chunk in sorted(timed, key=lambda c: float(c[1]["SampleTime"])):
        for when, tags in gps_fixes(chunk):
            if when < start_s:
                continue
            fix = _coordinate(tags, when)
            if fix is not None:
                return fix

    # The untimed container fix only answers a search that starts at 0
    if start_s > 0.0:
        return None
    return _coordinate(main, 0.0)


def _coordinate(tags: dict, when: float) -> dict | None:
    """
    The locked fix a tag set carries, or None when it holds no usable one.
    """
    lat, lon = tags.get("GPSLatitude"), tags.get("GPSLongitude")
    if lat is None or lon is None:
        return None
    if float(lat) == 0.0 and float(lon) == 0.0:
        return None
    return {"latitude": float(lat), "longitude": float(lon), "source_time": when}


def first_chunk_at_or_after(dump: list[dict], start_s: float) -> float | None:
    """
    Smallest GPMF chunk SampleTime at or after a source time.

    Args:
        dump: exiftool JSON output.
        start_s: source time, seconds.

    Returns:
        The chunk time in seconds, or None when there is none.
    """
    _, chunks = iter_documents(dump)
    later = [float(tags["SampleTime"]) for _, tags, _ in chunks if "SampleTime" in tags]
    later = [when for when in later if when >= start_s]
    return min(later) if later else None


def gps_payload(dump: list[dict], alignment: dict) -> dict:
    """
    GPS and gpmd-timing block of the sidecar.

    - `gps`: first locked fix inside the cut, else the first fix anywhere in the source
    - `gps_source_anchored`: True when a timed fix came from outside the known cut
    - `gpmd_residual_s`: injected track start minus the cut, since chunks are 1 Hz
    - an original aligns as the identity, so its whole file is the window

    Args:
        dump: exiftool JSON output of the source.
        alignment: {"offset_s", "r", "ok"} from align.

    Returns:
        {"gps", "gps_source_anchored", "gpmd_first_chunk_s", "gpmd_residual_s"}.
    """
    fix = first_fix(dump, start_s=alignment["offset_s"]) if alignment["ok"] else None
    anchored = fix is None
    if anchored:
        fix = first_fix(dump, start_s=0.0)

    # A container-only fix (phone clips) is untimed, so it is never source-anchored
    _, chunks = iter_documents(dump)
    timed = any("SampleTime" in chunk[1] and gps_fixes(chunk) for chunk in chunks)
    anchored = anchored and fix is not None and timed

    # Injected track starts on the first 1 Hz chunk at or after the cut
    first_chunk = first_chunk_at_or_after(dump, alignment["offset_s"])
    return {
        "gps": fix,
        "gps_source_anchored": anchored,
        "gpmd_first_chunk_s": first_chunk,
        "gpmd_residual_s": None if first_chunk is None else first_chunk - alignment["offset_s"],
    }


########
# Telemetry
########

# Wide GPMF tags: column prefix -> (exiftool tag, components per sample)
_GPMF_STREAMS = {
    "accl": ("Accelerometer", 3),
    "gyro": ("Gyroscope", 3),
    "grav": ("GravityVector", 3),
    "cori": ("CameraOrientation", 4),
    "iori": ("ImageOrientation", 4),
}
_AXES = {3: ("x", "y", "z"), 4: ("w", "x", "y", "z")}

# GPS arrives as parallel single-value tags, one per fix, zipped by expand_gpmf_parallel
_GPMF_GPS_TAGS = ("GPSLatitude", "GPSLongitude", "GPSAltitude", "GPSSpeed", "GPSSpeed3D")
_GPMF_GPS_AXES = ("lat", "lon", "alt", "speed2d", "speed3d")


def expand_gpmf(dump: list[dict], key: str, components: int) -> tuple[list[float], list[tuple[float, ...]]]:
    """
    Expand a wide GPMF tag into per-sample rows on the source time axis.

    - each chunk holds a flat run of N-tuples plus SampleTime and SampleDuration
    - samples are spread evenly across the chunk; GPMF has no per-sample time
    - non-numeric and ragged chunks are logged and skipped

    Args:
        dump: exiftool JSON output.
        key: exiftool tag name, e.g. "Accelerometer".
        components: values per sample.

    Returns:
        (times, values): source times in seconds and one tuple per sample.
    """
    times, values = [], []
    _, chunks = iter_documents(dump)
    for _, chunk_tags, _ in chunks:
        raw = chunk_tags.get(key)
        if raw is None:
            continue
        try:
            numbers = [float(x) for x in (raw if isinstance(raw, list) else str(raw).split())]
        except ValueError:
            # A dump read without -b holds a "(Binary data …)" placeholder
            logger.warning("skipping non-numeric %s chunk: %.60s", key, raw)
            continue
        if not numbers or len(numbers) % components:
            logger.warning("skipping ragged %s chunk: %d values for %d components", key, len(numbers), components)
            continue
        count = len(numbers) // components
        start = float(chunk_tags["SampleTime"])
        step = float(chunk_tags["SampleDuration"]) / count
        for i in range(count):
            times.append(start + i * step)
            values.append(tuple(numbers[i * components : (i + 1) * components]))
    return times, values


def expand_gpmf_parallel(dump: list[dict], keys: tuple[str, ...]) -> tuple[list[float], list[tuple[float | None, ...]]]:
    """
    Zip parallel single-value GPMF tags into one row per GPS fix.

    - covers the chunk's own fix and every sub-group fix of that second
    - an absent key yields None for that component
    - the untimed container document is never a sample

    Args:
        dump: exiftool JSON output.
        keys: exiftool tag names, one per component.

    Returns:
        (times, values): source times in seconds and one tuple per fix.
    """
    times, values = [], []
    _, chunks = iter_documents(dump)
    for chunk in chunks:
        if "SampleTime" not in chunk[1]:
            continue
        for when, tags in gps_fixes(chunk):
            row = tuple(None if tags.get(key) is None else float(tags[key]) for key in keys)
            if all(value is None for value in row):
                continue
            times.append(when)
            values.append(row)
    return times, values


def telemetry_table(dump: list[dict], offset_s: float | None, duration_s: float) -> pa.Table | None:
    """
    Per-sample telemetry on the union of every stream's sample times.

    - streams have disjoint clocks, so columns are sparse; null means no sample at that time
    - nothing is resampled; each column's fill ratio is logged
    - `edit_time` and `in_edit` are null when the cut window is unknown

    Args:
        dump: exiftool JSON output of the source.
        offset_s: cut start in source time, or None when alignment was rejected.
        duration_s: edit duration, seconds.

    Returns:
        A pyarrow table, or None when no stream produced data.
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

    if offset_s is not None:
        columns["edit_time"] = [t - offset_s for t in axis]
        end = offset_s + duration_s
        columns["in_edit"] = [offset_s <= t <= end for t in axis]
    else:
        columns["edit_time"] = [None] * len(axis)
        columns["in_edit"] = [None] * len(axis)

    table = pa.table(columns)

    # Log each column's fill ratio so sparsity is visible
    fills = [
        f"{name}={1.0 - table.column(name).null_count / table.num_rows:.0%}"
        for name in table.column_names
        if name != "source_time"
    ]
    logger.info("telemetry: %d rows on the union time axis, fill %s", table.num_rows, " ".join(fills))
    return table


def write_telemetry(table: pa.Table, dest_dir: Path, video: Path) -> Path:
    """
    Write the telemetry table beside the curated video.

    Args:
        table: from telemetry_table.
        dest_dir: curated folder.
        video: the curated video.

    Returns:
        `<dest_dir>/<stem>_telemetry.parquet`.
    """
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

# static_tags exiftool can write; LensProjection and EIS are read-only but stay in the sidecar
_WRITABLE_TAGS = frozenset(
    {
        "Make",
        "Model",
        "SerialNumber",
        "FirmwareVersion",
        "CreateDate",
        "MediaCreateDate",
        "FieldOfView",
        "ProjectionType",
        "GPSCoordinates",
        "GPSAltitude",
    }
)


def gpmd_command(
    curated: Path, source: Path, offset_s: float, duration_s: float, gpmd_index: int, out: Path
) -> list[str]:
    """
    ffmpeg call that grafts the trimmed gpmd track onto the curated video.

    - `-ss`/`-t` precede the second `-i`, so they trim the source
    - `-c copy` keeps the edit's pixels; `-copy_unknown` lets the bin_data track through
    - `-map -0:d` drops the export's `tmcd`, which the mp4 muxer cannot tag

    Args:
        curated: the curated edit.
        source: its camera original.
        offset_s: cut start in source time, seconds.
        duration_s: edit duration, seconds.
        gpmd_index: stream index of the source's gpmd track.
        out: output path.

    Returns:
        The argv list.
    """
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(curated),
        "-ss",
        f"{offset_s}",
        "-t",
        f"{duration_s}",
        "-i",
        str(source),
        "-map",
        "0",
        "-map",
        "-0:d",
        "-map",
        f"1:{gpmd_index}",
        "-c",
        "copy",
        "-copy_unknown",
        str(out),
    ]


def tag_command(curated: Path, tags: dict) -> list[str]:
    """
    exiftool call that writes static tags into the finished container.

    - only `_WRITABLE_TAGS` with a non-empty value are written

    Args:
        curated: the curated video.
        tags: from static_tags.

    Returns:
        The argv list.
    """
    # -n matches exif_dump's raw values; without it GPSCoordinates is silently dropped
    command = ["exiftool", "-overwrite_original", "-api", "QuickTimeUTC", "-n"]
    for key, value in tags.items():
        if value in (None, "") or key not in _WRITABLE_TAGS:
            continue
        command.append(f"-{key}={value}")
    command.append(f"-Software={PROVENANCE_TAG}")
    command.append(str(curated))
    return command


def _last_stderr_line(text: str) -> str:
    """
    Last non-empty stderr line, so a failure logs as one readable line.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "(no stderr)"


def inject(curated: Path, source: Path, offset_s: float | None, duration_s: float, tags: dict) -> bool:
    """
    Write metadata back into the curated video.

    - ffmpeg first (it rewrites the container), then exiftool writes tags into the result
    - the remux goes to a temp file moved into place only on success
    - exiftool warnings are logged; a non-zero exit from either tool raises
    - no gpmd is grafted when the cut window is unknown

    Args:
        curated: the curated edit.
        source: its camera original.
        offset_s: cut start in source time, or None when alignment was rejected.
        duration_s: edit duration, seconds.
        tags: from static_tags.

    Returns:
        True when a gpmd track landed.

    Raises:
        RuntimeError: ffmpeg or exiftool exited non-zero.
    """
    injected = False
    gpmd_index = find_gpmd_index(source) if offset_s is not None else None
    if offset_s is None:
        logger.warning("%s: no gpmd injected, alignment was rejected", curated.name)
    elif gpmd_index is None:
        logger.info("%s: no gpmd track in the source, writing static tags only", curated.name)
    else:
        # ".inject" goes before the extension: ffmpeg picks the muxer from the final suffix
        temp = curated.with_suffix(".inject" + curated.suffix)
        command = gpmd_command(curated, source, offset_s, duration_s, gpmd_index, temp)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            os.replace(temp, curated)
            injected = True
        else:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"gpmd injection failed for {curated.name}: {_last_stderr_line(result.stderr)}")

    # exiftool exits 0 when any tag lands, so dropped tags show only as stderr warnings
    tagged = subprocess.run(tag_command(curated, tags), capture_output=True, text=True, check=False)
    for line in tagged.stderr.splitlines():
        if "Warning:" in line or "Error" in line:
            logger.warning("static tags for %s: %s", curated.name, line.strip())
    if tagged.returncode != 0:
        raise RuntimeError(f"static tags failed for {curated.name}: {_last_stderr_line(tagged.stderr)}")
    return injected


########
# Pruning redundant src/ copies
########


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    SHA-256 of a file, read in chunks.

    Args:
        path: file to hash.
        chunk_size: bytes per read.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def is_redundant(video: Path, source: Path) -> bool:
    """
    Whether a `src/` copy is a bit-for-bit duplicate of the video above it.

    - size first, full hash only on equal sizes; nothing weaker authorizes a deletion
    - a same-duration re-encode differs in bytes and stays an edit

    Args:
        video: the top-level video.
        source: its `src/` counterpart.

    Returns:
        True when the two files are identical.
    """
    if video.stat().st_size != source.stat().st_size:
        return False
    return file_digest(video) == file_digest(source)


def prune_src(source_root: Path, only: str | None = None, apply: bool = False) -> int:
    """
    Report `src/` copies identical to the video above them; delete them when applied.

    - only the `src/` copy is deleted; the video then plans as an original
    - the match is re-checked right before each unlink
    - an emptied `src/` is removed; one still holding other files is kept

    Args:
        source_root: capture tree root.
        only: substring a flat name must contain, or None for all.
        apply: delete instead of report.

    Returns:
        Files deleted; 0 for a report.
    """
    candidates = 0
    candidate_bytes = 0
    removed = 0
    freed = 0
    for video, source, name in plan_videos(source_root):
        if source is None or (only and only not in name):
            continue
        if not is_redundant(video, source):
            continue
        candidates += 1
        size = source.stat().st_size
        candidate_bytes += size
        logger.info("redundant %s == %s (%.2f GB, sha256 %s)", source, video, size / 1e9, file_digest(source)[:16])
        if not apply:
            continue

        # Re-check right before the unlink; the Drive-synced tree may have changed
        if not is_redundant(video, source):
            logger.warning("%s changed since the scan, not deleted", source)
            continue
        src_dir = source.parent
        source.unlink()
        removed += 1
        freed += size
        logger.info("deleted %s", source)

        # Remove an emptied src/; one still holding anything else is kept
        if not find_videos(src_dir):
            try:
                src_dir.rmdir()
                logger.info("removed empty %s", src_dir)
            except OSError:
                logger.info("%s holds non-video files, kept", src_dir)

    if apply:
        logger.info("prune: %d deleted, %.2f GB freed", removed, freed / 1e9)
    else:
        logger.info(
            "prune: %d redundant copies, %.2f GB, nothing deleted (pass --apply)", candidates, candidate_bytes / 1e9
        )
    return removed


########
# Pipeline
########


def require_binaries() -> None:
    """
    Exit before any work when ffmpeg, ffprobe or exiftool is missing.

    Raises:
        SystemExit: naming the missing tools.
    """
    missing = [name for name in ("ffmpeg", "ffprobe", "exiftool") if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"missing required tools: {', '.join(missing)}")


def push(output_root: Path, dry_run: bool = False) -> None:
    """
    Run the rclone push script over the curated tree.

    Args:
        output_root: flat curated tree.
        dry_run: forward `--dry-run`.
    """
    command = [str(PUSH_SCRIPT), "--source", str(output_root)]
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


def process_video(video: Path, source: Path | None, name: str, output_root: Path, min_r: float, force: bool) -> dict:
    """
    Curate one video: copy, extract metadata, and for an edit align and inject.

    - an original is its own source: identity alignment, native gpmd, nothing grafted
    - a current copy with a stale sidecar has its metadata rewritten only

    Args:
        video: the video to curate.
        source: its camera original, or None for an original.
        name: flat folder name.
        output_root: flat curated tree.
        min_r: correlation needed to accept the alignment.
        force: re-copy and re-inject even when up to date.

    Returns:
        Summary dict with "name" and "status" (processed, refreshed or skipped).
    """
    dest_dir = output_root / name
    curated = dest_dir / video.name
    edited = source is not None

    # An original is its own source, so every metadata read below points at the same file
    meta_source = source if edited else video

    if needs_copy(video, dest_dir, force=force):
        copy_video(video, dest_dir, force=True)
        status = "processed"
    elif needs_refresh(video, source, dest_dir):
        # The copy is current but the sidecar no longer describes it: rewrite metadata alone
        logger.info("%s: refreshing stale metadata", name)
        status = "refreshed"
    else:
        logger.info("%s: up to date", name)
        return {"name": name, "status": "skipped"}

    alignment = align(video, source, name, min_r=min_r) if edited else {"offset_s": 0.0, "r": 1.0, "ok": True}
    dump = exif_dump(meta_source)
    duration_s = probe_duration(curated)

    # A rejected alignment leaves the cut window unknown, which downstream reads as None
    offset_s = alignment["offset_s"] if alignment["ok"] else None
    table = telemetry_table(dump, offset_s, duration_s)
    if table is not None:
        write_telemetry(table, dest_dir, video)

    payload = {
        "unique_id": name,
        "edit": str(video),
        "source_path": str(meta_source),
        "source": source_fingerprint(video),
        "edited": edited,
        "duration_s": duration_s,
        "has_imu": table is not None,
        "alignment": alignment,
        "static_tags": static_tags(dump),
        # Edits are cuts and color correction only, so lens geometry still holds
        "intrinsics": {"valid_for_edit": True, "reason": "cuts and colour correction only, no reframe"},
        **gps_payload(dump, alignment),
    }

    # Inject only into a freshly copied edit; a refresh keeps the recorded result
    if edited and status == "processed":
        payload["gpmd_injected"] = inject(curated, source, offset_s, duration_s, payload["static_tags"])
    elif edited:
        payload["gpmd_injected"] = read_metadata(dest_dir, video).get("gpmd_injected", False)
    else:
        # An original's gpmd track is native and never touched
        payload["gpmd_injected"], payload["gpmd_native"] = False, True
    write_metadata(dest_dir, video, payload)

    return {
        "name": name,
        "status": status,
        "aligned": alignment["ok"],
        "r": alignment["r"],
        "imu": table is not None,
        "injected": payload["gpmd_injected"],
    }


########
# Entry point
########


def main(argv: list[str] | None = None) -> int:
    """
    Parse args and run the curation pipeline.

    Args:
        argv: CLI arguments, or None for sys.argv.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("source_root", type=Path, nargs="?", default=DEFAULT_SOURCE_ROOT, help="capture tree to read")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="flat curated tree to write")
    parser.add_argument("--only", help="process only videos whose flat name contains this substring")
    parser.add_argument("--dry-run", action="store_true", help="log the plan and total size, change nothing")
    parser.add_argument("--force", action="store_true", help="re-copy and re-inject even when up to date")
    parser.add_argument("--index-only", action="store_true", help="rebuild index.csv from the sidecars and stop")
    parser.add_argument("--prune-src", action="store_true", help="report src/ copies identical to the video above")
    parser.add_argument("--apply", action="store_true", help="with --prune-src, actually delete what it reports")
    parser.add_argument("--push", action="store_true", help="run scripts/push_curated.sh after processing")
    parser.add_argument("--align-min-r", type=float, default=0.95, help="minimum correlation to accept")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # --index-only reads only sidecars, so it needs no media tools or Drive mount
    if args.index_only:
        write_index(args.output_root)
        return 0

    if not args.source_root.is_dir():
        parser.error(f"source root does not exist: {args.source_root}")

    # Pruning compares bytes and deletes files; it reads no media and needs no media tools
    if args.prune_src:
        prune_src(args.source_root, only=args.only, apply=args.apply)
        return 0

    require_binaries()

    entries = plan_videos(args.source_root)
    if args.only:
        entries = [entry for entry in entries if args.only in entry[2]]
    if not entries:
        if args.only:
            logger.warning("no videos matched --only %r under %s", args.only, args.source_root)
        else:
            logger.warning("no videos found under %s", args.source_root)
        return 0

    if args.dry_run:
        total = sum(video.stat().st_size for video, _, _ in entries)
        for video, source, name in entries:
            logger.info("%s/%s  <- %s", name, video.name, source or "(original)")
        logger.info("dry run: %d videos, %.1f GB", len(entries), total / 1e9)

        # A dry run that was also asked to push forwards the flag rather than dropping it
        if args.push:
            push(args.output_root, dry_run=True)
        return 0

    # One bad clip is recorded as failed so the index and summary still run
    records = []
    for video, source, name in entries:
        try:
            records.append(process_video(video, source, name, args.output_root, args.align_min_r, args.force))
        except Exception:
            logger.exception("%s: failed, skipping", name)
            records.append({"name": name, "status": "failed"})
    write_index(args.output_root)

    written = [r for r in records if r["status"] in ("processed", "refreshed")]
    failed = [r["name"] for r in records if r["status"] == "failed"]
    unaligned = [r["name"] for r in written if not r["aligned"]]
    logger.info(
        "done: %d processed, %d refreshed, %d skipped, %d failed, %d with IMU, %d injected",
        sum(1 for r in records if r["status"] == "processed"),
        sum(1 for r in records if r["status"] == "refreshed"),
        sum(1 for r in records if r["status"] == "skipped"),
        len(failed),
        sum(1 for r in written if r["imu"]),
        sum(1 for r in written if r["injected"]),
    )

    # Surfaced loudly: these clips carry source-time telemetry and no injected track
    for name in unaligned:
        logger.warning("alignment rejected, telemetry left in source time: %s", name)

    # Nothing was written for these at all, so they need a human
    for name in failed:
        logger.warning("processing failed, nothing curated: %s", name)

    if args.push:
        push(args.output_root, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
