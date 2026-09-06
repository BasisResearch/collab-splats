"""
Convert a scene's frames.zarr into images/ + frames.json with no video decode.

Existing processed scenes hold frames.zarr and no images/. Re-running preproc
would re-decode the source video (98 s cold per scene); this reads the store
instead. The old store is left in place — delete it once the scene reads back.

Usage:
    python scripts/migrate_frames_zarr.py <scene_dir> [<scene_dir> ...]
    python scripts/migrate_frames_zarr.py --all <processed_root>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import zarr

from collab_splats.preproc import frames as fr

logger = logging.getLogger(__name__)


def migrate_scene(scene_dir: Path) -> int:
    """
    Write images/ + frames.json from a scene's frames.zarr.

    Args:
        scene_dir: directory holding frames.zarr.

    Returns:
        Number of frames written.
    """
    scene_dir = Path(scene_dir)
    store_path = scene_dir / "frames.zarr"
    if not store_path.exists():
        raise FileNotFoundError(f"no frames.zarr in {scene_dir}")

    store = zarr.open(str(store_path), mode="r")
    images = store["images"][:]
    keys = list(store.attrs.get("record_keys", ["frame_idx"]))

    # Columnar arrays back into row dicts, one per selected frame
    columns = {k: store[k][:] for k in keys}
    records = [{k: columns[k][row] for k in keys} for row in range(images.shape[0])]

    provenance = dict(store.attrs.get("provenance", {}))
    fr.write_frames(scene_dir / "images", [np.asarray(f) for f in images], records, provenance)
    return int(images.shape[0])


def main(argv=None) -> int:
    """
    CLI entry point.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path, help="scene directories, or a root with --all")
    parser.add_argument("--all", action="store_true", help="treat each argument as a root of scene directories")
    args = parser.parse_args(argv)

    # --all expands each root into the scenes under it that still hold a store
    targets: list[Path] = []
    for arg in args.scenes:
        if args.all:
            targets += sorted(p.parent for p in Path(arg).glob("*/frames.zarr"))
        else:
            targets.append(Path(arg))

    for scene in targets:
        n = migrate_scene(scene)
        logger.info("migrated %s (%d frames)", scene, n)

    logger.info("migrated %d scene(s)", len(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
