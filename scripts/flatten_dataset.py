#!/usr/bin/env python3
"""Flatten the nested Google Drive capture tree into one folder per video.

Source layout (as pulled from Drive), where the parent folder name is not stable
across dates (``GoproSplat``, ``Goprosplat``, ``splats``, ``Phone pics and splat
videos``)::

    <source-root>/<YYYY-MM-DD>/<parent folder>/[videos]     <- processed
                                             /src/[videos]  <- camera originals, ignored

Only videos sitting *directly* inside ``<parent folder>`` are copied. A parent
folder holding nothing but a ``src/`` subfolder is skipped entirely.

Output layout — one flat folder per video, hyphen-delimited::

    <output-root>/<YYYY_MM_DD>-<parent>-<video stem>/<original filename>
    e.g.  2026_07_15-Goprosplat-GH010228/GH010228.mp4

Spaces, hyphens and other special characters inside the *parent folder* become
underscores, so the hyphen stays a reliable delimiter. The video stem is kept
verbatim, dots included — ``PXL_20260630_002106958.TS.mp4`` yields
``..-PXL_20260630_002106958.TS/``.

Usage:
    # Preview what would be copied, and how many bytes
    python scripts/flatten_dataset.py --dry-run

    # Do it (defaults: ../gdrive-src -> ../environments-curated)
    python scripts/flatten_dataset.py

    # Explicit roots
    python scripts/flatten_dataset.py \\
        --source-root /data/gdrive-src --output-root /data/environments-curated

Re-runs are cheap: a destination file that already exists with a matching size is
left alone unless --force is passed.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

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

_VIDEO_EXTS = {".mp4", ".mov", ".avi"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


########
# Helpers
########


def sanitize(name):
    """Replace runs of non-alphanumeric characters with a single underscore."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def find_videos(folder):
    """Return videos sitting directly inside `folder`, sorted; never recurses."""
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in _VIDEO_EXTS)


def flat_dir_name(date, parent, video):
    """Build the flat folder name YYYY_MM_DD-PARENT-VIDEO for one video.

    The video stem is carried over verbatim so the folder is traceable back to the
    original filename (e.g. PXL_20260630_002106958.TS keeps its dot). Only the parent
    folder is sanitized. The stem is last, so any hyphens it contains stay unambiguous
    when splitting the name on the first two hyphens.
    """
    return f"{date.replace('-', '_')}-{sanitize(parent)}-{video.stem}"


def plan_copies(source_root):
    """Walk <source-root>/<date>/<parent>/ and return (video, flat_dir_name) pairs.

    Parent folders with no directly-contained videos (i.e. `src`-only folders) are
    skipped. Raises ValueError if two videos would collide on the same flat name.
    """
    pairs = []
    # Top level must be dated dirs; anything else (stray files, .DS_Store) is ignored
    for date_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if not _DATE_RE.fullmatch(date_dir.name):
            logger.info("skip %s: not a YYYY-MM-DD directory", date_dir.name)
            continue
        for parent_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            videos = find_videos(parent_dir)
            if not videos:
                logger.info(
                    "skip %s/%s: no videos directly inside (src-only)",
                    date_dir.name,
                    parent_dir.name,
                )
                continue
            for video in videos:
                pairs.append((video, flat_dir_name(date_dir.name, parent_dir.name, video)))

    # A collision would silently overwrite one video with another
    seen = {}
    for video, name in pairs:
        if name in seen:
            raise ValueError(f"flat name collision {name!r}: {seen[name]} and {video}")
        seen[name] = video
    return pairs


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


########
# Entry point
########


def main(argv=None):
    """Parse args and flatten the source tree into the output root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"nested capture tree to read (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"flat tree to write (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--dry-run", action="store_true", help="log the plan and total size, copy nothing")
    parser.add_argument("--force", action="store_true", help="re-copy even when the destination exists")
    args = parser.parse_args(argv)

    if not args.source_root.is_dir():
        parser.error(f"source root does not exist: {args.source_root}")

    pairs = plan_copies(args.source_root)
    if not pairs:
        logger.warning("no videos found under %s", args.source_root)
        return 0

    # Dry run: show every destination folder and the total transfer size
    if args.dry_run:
        total = 0
        for video, name in pairs:
            size = video.stat().st_size
            total += size
            logger.info("%s/%s  <- %s (%.1f GB)", name, video.name, video, size / 1e9)
        logger.info("dry run: %d videos, %.1f GB total", len(pairs), total / 1e9)
        return 0

    copied = skipped = 0
    total = 0
    for video, name in pairs:
        size = copy_video(video, args.output_root / name, force=args.force)
        if size:
            copied += 1
            total += size
        else:
            skipped += 1

    logger.info(
        "done: %d copied (%.1f GB), %d already present -> %s",
        copied,
        total / 1e9,
        skipped,
        args.output_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
