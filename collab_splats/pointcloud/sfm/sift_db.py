"""
pycolmap SIFT database helpers shared by the sfm creators.

- SIFT extraction and matching run on GPU with the pycolmap-cuda12 wheel, else CPU
- used by the colmap and InstantSfM creators; not re-exported from sfm/__init__.py
- build_sift_database reimplements https://github.com/cre185/InstantSfM @ d3e599e,
  instantsfm/controllers/feature_handler.py:18-57 (GenerateDatabase)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import urllib.request
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pycolmap

from collab_splats.utils.torch_utils import get_device

logger = logging.getLogger(__name__)


########################################################################
# Constants
########################################################################


# Name of the pycolmap matching function for each pairing mode
_MATCHERS = {
    "sequential": "match_sequential",
    "retrieval": "match_vocabtree",
    "sequential+retrieval": "match_sequential",
    "exhaustive": "match_exhaustive",
}
PAIRINGS = tuple(_MATCHERS)

# COLMAP's vocab tree file for image retrieval, in the newer FAISS format
VOCAB_TREE_NAME = "vocab_tree_faiss_flickr100K_words32K.bin"  # pycolmap 4 crashes on the older flann file
VOCAB_TREE_URL = (
    f"https://github.com/colmap/colmap/releases/download/3.11.1/{VOCAB_TREE_NAME}"
)
VOCAB_TREE_SHA256 = "921e894b7d81f5cf223df824a02b9932660cddf00a815c93fc7c0bd690fc639e"
VOCAB_TREE_CACHE = Path.home() / ".cache" / "collab_splats"


########################################################################
# SIFT feature database
########################################################################


def build_sift_database(
    image_path: Path,
    database_path: Path,
    *,
    pairing: str,
    overlap: int,
    num_retrieved: int,
    vocab_tree: Path | None,
    num_threads: int,
) -> None:
    """
    Build the SIFT feature database with pycolmap: extraction, then matching under one pairing mode.

    - all images share one SIMPLE_RADIAL camera
    - sequential pairs each image with the next `overlap` images, no quadratic overlap
    - on failure the partial DB is deleted

    Args:
        image_path: directory of images to extract from.
        database_path: SIFT database to write.
        pairing: which image pairs are matched; one of PAIRINGS.
        overlap: sequential neighbors matched per image.
        num_retrieved: vocab-tree neighbors retrieved per image.
        vocab_tree: vocab tree file; required when pairing retrieves (see fetch_vocab_tree).
        num_threads: CPU thread cap; GPU extraction and matching ignore it.

    Raises:
        ValueError: unknown pairing, or a retrieval pairing without vocab_tree.
        RuntimeError: pycolmap extraction or matching fails.
    """
    # Check the arguments before calling pycolmap, so bad values leave no database behind
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")

    if "retrieval" in pairing and vocab_tree is None:
        raise ValueError(
            f"pairing {pairing!r} needs a vocab_tree (see fetch_vocab_tree)"
        )

    # Set the pairing options for the chosen mode
    if pairing == "retrieval":
        pairing_options = pycolmap.VocabTreePairingOptions(
            num_images=num_retrieved, vocab_tree_path=vocab_tree
        )
    elif pairing == "exhaustive":
        pairing_options = pycolmap.ExhaustivePairingOptions()
    else:
        pairing_options = pycolmap.SequentialPairingOptions(
            overlap=overlap, quadratic_overlap=False
        )

        if pairing == "sequential+retrieval":
            pairing_options.loop_detection = True
            pairing_options.loop_detection_num_images = num_retrieved
            pairing_options.vocab_tree_path = vocab_tree

    # Use the GPU only if pycolmap was built with CUDA and a GPU is available
    use_gpu = pycolmap.has_cuda and get_device() == "cuda"
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    # Limit the number of CPU threads
    if "retrieval" in pairing:
        pairing_options.num_threads = (
            num_threads  # uncapped, OpenBLAS can crash on many cores
        )

    extraction_options = pycolmap.FeatureExtractionOptions()
    matching_options = pycolmap.FeatureMatchingOptions()

    if not use_gpu:
        extraction_options.num_threads = num_threads
        matching_options.num_threads = num_threads

    # Extract features with one shared camera, then match, and delete the partial database on failure
    try:
        logger.info(
            "SIFT database %s: extract + %s (%s)",
            database_path,
            _MATCHERS[pairing],
            device.name,
        )
        pycolmap.extract_features(
            database_path,
            image_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL"),
            extraction_options=extraction_options,
            device=device,
        )
        getattr(pycolmap, _MATCHERS[pairing])(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    except (RuntimeError, ValueError) as err:
        database_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"COLMAP SIFT database build failed on {device.name} ({err})"
        ) from err


def ensure_sift_database(
    image_dir: Path,
    db_path: Path,
    names: list[str],
    *,
    pairing: str,
    overlap: int = 10,
    num_retrieved: int = 20,
    vocab_tree: Path | Callable[[], Path] | None = None,
    num_threads: int,
) -> None:
    """
    SIFT database for names, reused when its images and stored params still match.

    - the params live in a collab_params table inside the DB
    - only params the pairing mode reads are compared
    - a mismatch, missing params or an incomplete DB triggers a full rebuild

    Args:
        image_dir: directory holding names.
        db_path: database file to reuse or build.
        names: image filenames the DB must hold.
        pairing: sequential, retrieval, sequential+retrieval or exhaustive.
        overlap: sequential window, read by sequential modes.
        num_retrieved: neighbors per image, read by retrieval modes.
        vocab_tree: vocab tree file for retrieval modes, or a callable resolved only on a rebuild.
        num_threads: CPU thread cap; see build_sift_database.

    Raises:
        ValueError: see build_sift_database.
        RuntimeError: the rebuild fails.
    """
    # Collect the settings this pairing mode uses, to compare with the stored ones
    params: dict[str, str | int] = {"pairing": pairing}

    if pairing.startswith("sequential"):
        params["overlap"] = overlap

    if "retrieval" in pairing:
        params["num_retrieved"] = num_retrieved

    # Reuse the existing database if it has the same images and settings
    if _database_holds(db_path, names) and _stored_params(db_path) == params:
        logger.info("SIFT database %s: reusing", db_path)

        return

    # Otherwise delete it and rebuild from scratch
    db_path.unlink(missing_ok=True)
    logger.info("SIFT database %s: building (%s)", db_path, pairing)
    build_sift_database(
        image_dir,
        db_path,
        pairing=pairing,
        overlap=overlap,
        num_retrieved=num_retrieved,
        vocab_tree=vocab_tree() if callable(vocab_tree) else vocab_tree,
        num_threads=num_threads,
    )

    # Save the settings in the database so the next call can check them
    row = json.dumps(params, sort_keys=True)
    conn = sqlite3.connect(db_path)

    with closing(conn), conn:
        conn.execute("CREATE TABLE IF NOT EXISTS collab_params (json TEXT)")
        conn.execute("INSERT INTO collab_params VALUES (?)", (row,))


########################################################################
# Vocab tree
########################################################################


def fetch_vocab_tree(cache_dir: Path | None = None) -> Path:
    """
    Path to the pinned COLMAP vocab tree, downloaded once into the cache.

    - sha256-checked on every call; a mismatched file is deleted, never used

    Args:
        cache_dir: where the file lives; defaults to ~/.cache/collab_splats.

    Returns:
        The local vocab tree path.

    Raises:
        RuntimeError: the download fails or the sha256 does not match.
    """
    cache_dir = Path(cache_dir or VOCAB_TREE_CACHE)
    path = cache_dir / VOCAB_TREE_NAME

    # Download the file if it is not cached yet
    if not path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("colmap: downloading %s -> %s", VOCAB_TREE_URL, path)

        try:
            urllib.request.urlretrieve(VOCAB_TREE_URL, path)
        except OSError as err:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"vocab tree download failed ({err}): {VOCAB_TREE_URL} -> {path}"
            ) from err

    # Check the file hash, and delete the file if it does not match
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    if digest != VOCAB_TREE_SHA256:
        path.unlink()
        raise RuntimeError(
            f"vocab tree sha256 {digest} != pinned {VOCAB_TREE_SHA256} ({VOCAB_TREE_URL}, {path})"
        )

    return path


########################################################################
# Private helpers
########################################################################


def _database_holds(database_path: Path, image_names: list[str]) -> bool:
    """
    True when the SIFT DB holds extraction and matching output for exactly `image_names`.

    - False when the DB is missing, unreadable, partial, or holds a different image set
    - image_names in any order
    """
    # Check the file exists first, since opening a missing database creates it
    if not database_path.exists():
        return False

    try:
        db = pycolmap.Database.open(str(database_path))
    except RuntimeError:
        return False

    # Check the image names match and the database holds features and matches
    try:
        registered = {image.name for image in db.read_all_images()}

        if registered != set(image_names):
            logger.info(
                "SIFT database %s holds %d images against this run's %d — rebuilding",
                database_path,
                len(registered),
                len(image_names),
            )

            return False

        return db.num_keypoints() > 0 and db.num_verified_image_pairs() > 0
    finally:
        db.close()


def _stored_params(database_path: Path) -> dict | None:
    """
    Matching params the DB was built with, or None for a DB without a params row.

    - None also when the collab_params table is missing
    """
    conn = sqlite3.connect(database_path)

    with closing(conn):
        try:
            row = conn.execute("SELECT json FROM collab_params").fetchone()
        except sqlite3.OperationalError:
            return None

    return None if row is None else json.loads(row[0])
