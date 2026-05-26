"""Download 7-Scenes sequences from the Microsoft Research CDN.

Usage::

    python evals/download_7scenes.py --scenes fire office
    python evals/download_7scenes.py --list
    python evals/download_7scenes.py --scenes chess --seq seq-02 --force

Data is written to ``<repo_root>/data/7scenes/<scene>/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

########################################################################
# Constants
########################################################################

# Two levels up from this file (evals/ → repo root)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

_CDN = "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

# All seven scenes and their CDN zip URLs
SCENES: dict[str, str] = {
    "chess": f"{_CDN}/chess.zip",
    "fire": f"{_CDN}/fire.zip",
    "heads": f"{_CDN}/heads.zip",
    "office": f"{_CDN}/office.zip",
    "pumpkin": f"{_CDN}/pumpkin.zip",
    "redkitchen": f"{_CDN}/redkitchen.zip",
    "stairs": f"{_CDN}/stairs.zip",
}

########################################################################
# Helpers
########################################################################


def _scene_already_downloaded(scene_dir: Path, seq: str) -> bool:
    """Return True if ``scene_dir/seq/`` exists and contains at least one ``*.color.png``."""
    seq_dir = scene_dir / seq
    if not seq_dir.is_dir():
        return False
    return any(seq_dir.glob("*.color.png"))


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple download progress indicator."""
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded * 100 // total_size
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    logger.info("\r  [%s] %3d%%  %d/%d MB", bar, pct, downloaded // 1_048_576, total_size // 1_048_576)


########################################################################
# Core download function
########################################################################


def download_scene(scene: str, seq: str = "seq-01", force: bool = False) -> Path:
    """Download and extract one 7-Scenes sequence.

    Args:
        scene: Scene name (must be a key in ``SCENES``).
        seq:   Sequence directory name, e.g. ``"seq-01"``.
        force: Re-download even if the sequence is already present.

    Returns:
        Path to the extracted sequence directory.

    Raises:
        ValueError: If *scene* is not in ``SCENES``.
        RuntimeError: If extraction yields no ``*.color.png`` files.
    """
    if scene not in SCENES:
        raise ValueError(f"Unknown scene '{scene}'. Choose from: {sorted(SCENES)}")

    # Destination paths
    data_root = REPO_ROOT / "data" / "7scenes"
    scene_dir = data_root / scene
    seq_dir = scene_dir / seq

    # Skip if already present
    if not force and _scene_already_downloaded(scene_dir, seq):
        logger.info("  %s/%s already downloaded — skipping (use --force to re-download).", scene, seq)
        return seq_dir

    # Download zip to a temp location inside data/
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / f"{scene}.zip"
    url = SCENES[scene]

    logger.info("Downloading %s from %s", scene, url)
    urllib.request.urlretrieve(url, zip_path, reporthook=_progress_hook)

    # Extract — Microsoft's outer zip nests as <scene>/<scene>/seq-NN.zip
    logger.info("  Extracting %s ...", zip_path.name)
    scene_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as outer:
        outer.extractall(data_root)

    # Flatten nested wrapper: outer zip may produce <scene>/<scene>/ or just <scene>/
    # Find where the seq-NN.zip files actually landed
    candidate_dirs = [
        scene_dir / scene,  # doubly nested: data/7scenes/<scene>/<scene>/
        scene_dir,          # flat:          data/7scenes/<scene>/
    ]
    inner_dir: Path | None = None
    for candidate in candidate_dirs:
        if candidate.is_dir() and any(candidate.glob("seq-*.zip")):
            inner_dir = candidate
            break

    if inner_dir is not None and inner_dir != scene_dir:
        # Move contents up one level to flatten the wrapper
        for item in list(inner_dir.iterdir()):
            dest = scene_dir / item.name
            if not dest.exists():
                item.rename(dest)
        inner_dir.rmdir()

    # Extract each inner seq-NN.zip
    for inner_zip in sorted(scene_dir.glob("seq-*.zip")):
        seq_name = inner_zip.stem  # e.g. "seq-01"
        target_seq = scene_dir / seq_name
        target_seq.mkdir(exist_ok=True)
        logger.info("  Extracting %s ...", inner_zip.name)
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(target_seq)
        inner_zip.unlink()

    # Remove outer zip to save disk space
    zip_path.unlink(missing_ok=True)

    # Verify extraction succeeded
    if not list(seq_dir.glob("*.color.png")):
        raise RuntimeError(
            f"Extraction completed but no *.color.png files found in {seq_dir}. "
            "The archive layout may have changed."
        )

    logger.info("  Done. Data at %s", seq_dir)
    return seq_dir


########################################################################
# CLI
########################################################################


def main(argv: list[str] | None = None) -> None:
    """Entry point for the download CLI."""
    parser = argparse.ArgumentParser(
        description="Download 7-Scenes sequences from Microsoft Research CDN."
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=list(SCENES.keys()),
        default=["fire", "office"],
        metavar="SCENE",
        help=f"Scenes to download. Choices: {', '.join(sorted(SCENES))}. Default: fire office.",
    )
    parser.add_argument(
        "--seq",
        default="seq-01",
        help="Sequence to download (default: seq-01).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the sequence is already present.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print scene names and CDN URLs, then exit.",
    )

    args = parser.parse_args(argv)

    # --list mode: print scenes and exit
    if args.list:
        print("Available 7-Scenes scenes:")
        for name, url in sorted(SCENES.items()):
            print(f"  {name:<12} {url}")
        sys.exit(0)

    # Download each requested scene
    for scene in args.scenes:
        print(f"\n=== {scene} ===")
        download_scene(scene, seq=args.seq, force=args.force)


if __name__ == "__main__":
    main()
