"""Download evaluation datasets — one subcommand per dataset.

Usage::

    python evals/data/download_datasets.py 7scenes --scenes fire office
    python evals/data/download_datasets.py 7scenes --list
    python evals/data/download_datasets.py 7scenes --parity
    python evals/data/download_datasets.py co3dv2 apple ./evals/data/co3dv2
    python evals/data/download_datasets.py kitti 00 ./evals/data/kitti
    python evals/data/download_datasets.py tum desk /workspace/collab-splats/evals/data/tum
    python evals/data/download_datasets.py waymo <segment_id> <dest_dir>

Each dataset keeps its original output-dir convention (they differ on purpose):

    7scenes   → <repo_root>/data/7scenes/<scene>/           (urllib download + zip extract)
    7scenes --parity → evals/data/7scenes/<scene>/<scene>/seq-01 + evals/data/tum/ (wget/unzip/tar)
    co3dv2    → ./evals/data/co3dv2/<category>/              (co3d download_dataset.py)
    kitti     → registration-gated guide + layout verifier
    tum       → /workspace/collab-splats/evals/data/tum/<name>/  (wget + tar)
    waymo     → license-gated guide + layout verifier (extract via evals/data/extract_waymo.py)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

########################################################################
# Constants
########################################################################

# Three levels up from this file (evals/data/ → evals/ → repo root)
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Directory this script lives in (evals/data/); parity scenes anchor here,
# matching the old download_parity_scenes.sh `cd "$(dirname "$0")"`.
DATA_DIR: Path = Path(__file__).resolve().parent

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
# 7-Scenes — helpers
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
# 7-Scenes — core download
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
        scene_dir,  # flat:          data/7scenes/<scene>/
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

    # Extract each inner seq-NN.zip directly to scene_dir; the zip's top-level
    # dir is already named seq-NN, so this lands at scene_dir/seq-NN/frame-*.
    for inner_zip in sorted(scene_dir.glob("seq-*.zip")):
        logger.info("  Extracting %s ...", inner_zip.name)
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(scene_dir)
        inner_zip.unlink()

    # Remove outer zip to save disk space
    zip_path.unlink(missing_ok=True)

    # Verify extraction succeeded
    if not list(seq_dir.glob("*.color.png")):
        raise RuntimeError(
            f"Extraction completed but no *.color.png files found in {seq_dir}. " "The archive layout may have changed."
        )

    logger.info("  Done. Data at %s", seq_dir)
    return seq_dir


########################################################################
# 7-Scenes — LC-parity scene set (folds in download_parity_scenes.sh)
########################################################################

# Parity 7-Scenes subset (chess is assumed already local) — spec 2026-07-08-lc-parity-validation-design.
_PARITY_SEVEN_SCENES = ["fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

# Parity TUM sequences: <freiburgN>/<archive_name> -> short label. Spans fr1/fr2/fr3.
_PARITY_TUM = {
    "freiburg1/rgbd_dataset_freiburg1_desk": "fr1_desk",
    "freiburg1/rgbd_dataset_freiburg1_room": "fr1_room",
    "freiburg2/rgbd_dataset_freiburg2_xyz": "fr2_xyz",
    "freiburg3/rgbd_dataset_freiburg3_long_office_household": "fr3_office",
}
_PARITY_TUM_BASE = "https://cvg.cit.tum.de/rgbd/dataset"


def _unzip_ok(args: list[str]) -> None:
    """Run ``unzip -q -o``; tolerate exit 1 (7-Scenes STORED size-mismatch warning), fail on >=2."""
    rc = subprocess.run(["unzip", "-q", "-o", *args]).returncode
    if rc >= 2:
        raise subprocess.CalledProcessError(rc, "unzip")


def download_parity_scenes() -> None:
    """Download the LC-parity validation scenes (7-Scenes subset + TUM). Idempotent (~25 GB total)."""
    # 7-Scenes subset → evals/data/7scenes/<scene>/<scene>/seq-01
    for scene in _PARITY_SEVEN_SCENES:  # chess already local
        scene_out = DATA_DIR / "7scenes" / scene
        if (scene_out / scene / "seq-01").is_dir():
            print(f"[skip] 7scenes/{scene}")
            continue
        scene_out.mkdir(parents=True, exist_ok=True)
        print(f"[get ] 7-Scenes {scene}")
        zip_path = DATA_DIR / "7scenes" / f"{scene}.zip"
        subprocess.run(["wget", "-c", f"{_CDN}/{scene}.zip", "-O", str(zip_path)], check=True)
        _unzip_ok([str(zip_path), "-d", str(scene_out)])
        # scene zips contain per-seq zips; extract seq-01 only (parity uses seq-01)
        _unzip_ok([str(scene_out / scene / "seq-01.zip"), "-d", str(scene_out / scene)])

    # TUM subset → evals/data/tum/<name>
    tum_dir = DATA_DIR / "tum"
    tum_dir.mkdir(parents=True, exist_ok=True)
    for path, label in _PARITY_TUM.items():
        name = Path(path).name
        if (tum_dir / name).is_dir():
            print(f"[skip] tum/{name}")
            continue
        print(f"[get ] TUM {label}")
        tgz = tum_dir / f"{name}.tgz"
        subprocess.run(["wget", "-c", f"{_PARITY_TUM_BASE}/{path}.tgz", "-O", str(tgz)], check=True)
        subprocess.run(["tar", "-xzf", str(tgz), "-C", str(tum_dir)], check=True)

    print("Done. Verify: ls 7scenes/*/*/seq-01 | head; ls tum/")


########################################################################
# 7-Scenes — dispatch
########################################################################


def download_7scenes(
    scenes: list[str],
    seq: str = "seq-01",
    force: bool = False,
    parity: bool = False,
    list_scenes: bool = False,
) -> None:
    """Download 7-Scenes sequences (or --list / --parity variants)."""
    # --list mode: print scenes and exit
    if list_scenes:
        print("Available 7-Scenes scenes:")
        for name, url in sorted(SCENES.items()):
            print(f"  {name:<12} {url}")
        sys.exit(0)

    # --parity mode: the LC-parity validation set (7-Scenes subset + TUM)
    if parity:
        download_parity_scenes()
        return

    # Download each requested scene
    for scene in scenes:
        print(f"\n=== {scene} ===")
        download_scene(scene, seq=seq, force=force)


########################################################################
# CO3Dv2
########################################################################


def download_co3dv2(category: str, output_dir: str = "./evals/data/co3dv2") -> None:
    """Download a single CO3Dv2 category via the co3d package's download_dataset.py."""
    if not category:
        print("Usage: python download_datasets.py co3dv2 <category> [output_dir]")
        print("Categories: apple ball banana bench book bottle bowl broccoli car chair")
        sys.exit(1)

    standard_categories = "apple ball banana bench book bottle bowl broccoli car chair"
    if category not in standard_categories.split():
        print(f"Warning: '{category}' is not in the standard 10-category eval set.")
        print(f"Standard categories: {standard_categories}")

    print(f"=== Downloading CO3Dv2 category: {category} → {output_dir} ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Locate the co3d package's downloader in the reconstruction env (or $PYTHON)
    python = os.environ.get("PYTHON", "/opt/venv/reconstruction/bin/python")
    if subprocess.run([python, "-c", "import co3d"], stderr=subprocess.DEVNULL).returncode != 0:
        print("Error: 'co3d' package not found. Install with:")
        print(f"  {python} -m pip install co3d")
        print("  OR: git clone https://github.com/facebookresearch/co3d && pip install -e co3d/")
        sys.exit(1)

    co3d_script = subprocess.run(
        [python, "-c", 'import co3d, os; print(os.path.join(os.path.dirname(co3d.__file__), "download_dataset.py"))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    subprocess.run(
        [
            python,
            co3d_script,
            "--download_folder",
            output_dir,
            "--download_categories",
            category,
            "--single_sequence_subset",
        ],
        check=True,
    )

    print(f"Download complete: {output_dir}/{category}/")
    print("=== Done. Test with: ===")
    print("python evals/scripts/eval.py --dataset co3dv2 \\")
    print(f"  --seq_dir {output_dir}/{category}/<sequence_name> \\")
    print(f"  --output_dir ./eval_results/co3dv2_{category} \\")
    print("  --conditions baseline ba ba_hightrack")


########################################################################
# KITTI Odometry (registration-gated: guide + verifier)
########################################################################


def download_kitti(sequence: str = "00", dest: str = "./evals/data/kitti") -> None:
    """Verify a KITTI Odometry sequence layout, or print the (post-login) download guide."""
    cases = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    if sequence not in cases:
        print(f"Unknown sequence '{sequence}'. KITTI Odometry sequences with GT poses: {' '.join(cases)}")
        sys.exit(1)

    img_dir = Path(dest) / "sequences" / sequence / "image_2"
    poses_local = Path(dest) / "sequences" / sequence / "poses.txt"
    poses_alt = Path(dest) / "poses" / f"{sequence}.txt"

    # Already populated → report and exit 0
    if img_dir.is_dir() and any(img_dir.glob("*.png")):
        if poses_local.is_file() or poses_alt.is_file():
            print(f"OK: KITTI sequence {sequence} already populated under {dest}.")
            print(f"  Images: {img_dir}")
            if poses_local.is_file():
                print(f"  Poses:  {poses_local}")
            else:
                print(f"  Poses:  {poses_alt}")
            sys.exit(0)

    # Missing/incomplete → print canonical (post-login) download instructions
    print(f"""KITTI Odometry data missing or incomplete for sequence {sequence}.

KITTI requires (free) registration. Sign in at:
    https://www.cvlibs.net/datasets/kitti/user_login.php

Then fetch the post-login direct downloads (URLs are stable once authenticated):
    data_odometry_color.zip   (~65 GB, color images — needed by loader)
    data_odometry_gray.zip    (~22 GB, grayscale; optional)
    data_odometry_calib.zip   (small, calibration)
    data_odometry_poses.zip   (small, GT poses for sequences 00-10)

Example (replace <COOKIE> with your authenticated session cookie):
    wget --header="Cookie: <COOKIE>" \\
        https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip
    wget --header="Cookie: <COOKIE>" \\
        https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
    wget --header="Cookie: <COOKIE>" \\
        https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip

Extract all three under DEST={dest}. Expected post-extraction tree:
    {dest}/sequences/{sequence}/image_2/000000.png
    {dest}/sequences/{sequence}/image_2/000001.png
    ...
    {dest}/sequences/{sequence}/calib.txt
    {dest}/poses/{sequence}.txt
        (or equivalently: {dest}/sequences/{sequence}/poses.txt — loader accepts both)

Re-run this script after extraction to verify.""")
    sys.exit(1)


########################################################################
# TUM RGB-D (Freiburg-1)
########################################################################


def download_tum(seq: str = "desk", dest: str = "/workspace/collab-splats/evals/data/tum") -> None:
    """Download and extract a single TUM RGB-D Freiburg-1 sequence."""
    valid = ["desk", "desk2", "360", "floor", "plant", "room", "rpy", "teddy", "xyz"]
    if seq not in valid:
        print(f"Unknown sequence '{seq}'. Choose from: {' '.join(valid)}")
        sys.exit(1)

    name = f"rgbd_dataset_freiburg1_{seq}"
    url = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg1/{name}.tgz"

    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted (handle both flat and nested layouts)
    if (dest_p / name).is_dir() and (dest_p / name / "groundtruth.txt").is_file():
        print(f"Already extracted: {dest_p / name}")
        return
    if (dest_p / seq).is_dir() and (dest_p / seq / "groundtruth.txt").is_file():
        print(f"Already extracted: {dest_p / seq}")
        return

    tgz = dest_p / f"{name}.tgz"
    print(f"Downloading {seq} -> {tgz}")
    subprocess.run(["wget", "-c", url, "-O", str(tgz)], check=True)
    print("Extracting...")
    subprocess.run(["tar", "-xzf", str(tgz), "-C", str(dest_p)], check=True)
    print(f"Done. Data at {dest_p / name}/")
    print("")
    print("Expected structure:")
    print(f"  {dest_p / name}/rgb/*.png")
    print(f"  {dest_p / name}/rgb.txt")
    print(f"  {dest_p / name}/depth.txt")
    print(f"  {dest_p / name}/groundtruth.txt")


########################################################################
# Waymo Open Dataset (license-gated: guide + verifier)
########################################################################


def download_waymo(segment_id: str, dest: str = "/workspace/collab-splats/evals/data/waymo") -> None:
    """Verify a Waymo segment layout, or print the license-gated download + extraction guide."""
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    target = dest_p / segment_id

    # Already extracted → report and exit 0
    if (target / "groundtruth.txt").is_file() and (target / "images").is_dir():
        print(f"Waymo segment '{segment_id}' already extracted at {target}")
        sys.exit(0)

    print(f"""Waymo Open Dataset is license-gated. To prepare segment '{segment_id}':

  1. Register at https://waymo.com/open/ and accept the dataset terms.
  2. Download the v1.4.1 segment .tfrecord, e.g. via gsutil:

       gsutil cp gs://waymo_open_dataset_v_1_4_1/individual_files/training/{segment_id}.tfrecord \\
                "{dest}/{segment_id}.tfrecord"

  3. Extract with the sidecar script (in a python env that has
     waymo-open-dataset installed — TensorFlow conflicts with our torch
     stack, keep it isolated):

       python evals/data/extract_waymo.py \\
           --tfrecord "{dest}/{segment_id}.tfrecord" \\
           --output   "{target}" \\
           --camera   FRONT

  4. The extracted layout is what _load_waymo consumes.

Re-run this script after extraction; it will exit 0 once the expected
layout is present at {target}.""")
    sys.exit(1)


########################################################################
# CLI
########################################################################


def main(argv: list[str] | None = None) -> None:
    """Entry point: dispatch to a per-dataset downloader via subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Download evaluation datasets (choose a dataset subcommand).")
    sub = parser.add_subparsers(dest="dataset", required=True, metavar="dataset")

    # 7scenes
    p7 = sub.add_parser("7scenes", help="Download 7-Scenes sequences from Microsoft Research CDN.")
    p7.add_argument(
        "--scenes",
        nargs="+",
        choices=list(SCENES.keys()),
        default=["fire", "office"],
        metavar="SCENE",
        help=f"Scenes to download. Choices: {', '.join(sorted(SCENES))}. Default: fire office.",
    )
    p7.add_argument("--seq", default="seq-01", help="Sequence to download (default: seq-01).")
    p7.add_argument("--force", action="store_true", help="Re-download even if the sequence is already present.")
    p7.add_argument("--list", action="store_true", help="Print scene names and CDN URLs, then exit.")
    p7.add_argument(
        "--parity",
        action="store_true",
        help="Download the LC-parity validation set (7-Scenes subset + TUM) into evals/data/.",
    )

    # co3dv2
    pco = sub.add_parser("co3dv2", help="Download a single CO3Dv2 category.")
    pco.add_argument("category", nargs="?", default="", help="Category, e.g. apple ball banana ...")
    pco.add_argument(
        "output_dir", nargs="?", default="./evals/data/co3dv2", help="Destination (default: ./evals/data/co3dv2)."
    )

    # kitti
    pk = sub.add_parser("kitti", help="Guide + layout verifier for a KITTI Odometry sequence.")
    pk.add_argument("sequence", nargs="?", default="00", help="Two-digit sequence id, e.g. 00 02 05 06 07 09.")
    pk.add_argument("dest", nargs="?", default="./evals/data/kitti", help="Root dir (default: ./evals/data/kitti).")

    # tum
    pt = sub.add_parser("tum", help="Download a single TUM RGB-D Freiburg-1 sequence.")
    pt.add_argument("seq", nargs="?", default="desk", help="Sequence: desk desk2 360 floor plant room rpy teddy xyz.")
    pt.add_argument(
        "dest",
        nargs="?",
        default="/workspace/collab-splats/evals/data/tum",
        help="Destination (default: /workspace/collab-splats/evals/data/tum).",
    )

    # waymo
    pw = sub.add_parser("waymo", help="Guide + layout verifier for a Waymo segment (license-gated).")
    pw.add_argument("segment_id", help="Waymo segment id.")
    pw.add_argument(
        "dest",
        nargs="?",
        default="/workspace/collab-splats/evals/data/waymo",
        help="Destination (default: /workspace/collab-splats/evals/data/waymo).",
    )

    args = parser.parse_args(argv)

    # Dispatch to the selected dataset
    if args.dataset == "7scenes":
        download_7scenes(args.scenes, seq=args.seq, force=args.force, parity=args.parity, list_scenes=args.list)
    elif args.dataset == "co3dv2":
        download_co3dv2(args.category, args.output_dir)
    elif args.dataset == "kitti":
        download_kitti(args.sequence, args.dest)
    elif args.dataset == "tum":
        download_tum(args.seq, args.dest)
    elif args.dataset == "waymo":
        download_waymo(args.segment_id, args.dest)


if __name__ == "__main__":
    main()
