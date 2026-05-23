#!/usr/bin/env bash
# Guide + verifier for a single KITTI Odometry sequence.
#
# KITTI Odometry requires registration; bulk zips are NOT freely scriptable.
# This script prints the canonical download URLs (post-login) and verifies
# the local filesystem layout the loader (`evals/datasets.py:_load_kitti`)
# expects.
#
# Usage: bash evals/download_kitti.sh <sequence_NN> <dest_dir>
#   <sequence_NN>: two-digit sequence id, e.g. 00 02 05 06 07 09
#   <dest_dir>:    root dir; expected to contain `sequences/<NN>/image_2/`
#                  and either `sequences/<NN>/poses.txt` or `poses/<NN>.txt`.
#
# Sequences with GT loop closure (the VGGT-Long benchmark): 00 02 05 06 07 09.
set -euo pipefail

SEQ="${1:-00}"
DEST="${2:-./evals/data/kitti}"

CASES=(00 01 02 03 04 05 06 07 08 09 10)
ok=0
for c in "${CASES[@]}"; do [[ "$SEQ" == "$c" ]] && ok=1; done
if [[ "$ok" -ne 1 ]]; then
    echo "Unknown sequence '$SEQ'. KITTI Odometry sequences with GT poses: ${CASES[*]}"
    exit 1
fi

IMG_DIR="$DEST/sequences/$SEQ/image_2"
POSES_LOCAL="$DEST/sequences/$SEQ/poses.txt"
POSES_ALT="$DEST/poses/$SEQ.txt"

if [[ -d "$IMG_DIR" ]] && compgen -G "$IMG_DIR/*.png" > /dev/null; then
    if [[ -f "$POSES_LOCAL" || -f "$POSES_ALT" ]]; then
        echo "OK: KITTI sequence $SEQ already populated under $DEST."
        echo "  Images: $IMG_DIR"
        if [[ -f "$POSES_LOCAL" ]]; then
            echo "  Poses:  $POSES_LOCAL"
        else
            echo "  Poses:  $POSES_ALT"
        fi
        exit 0
    fi
fi

cat <<EOF
KITTI Odometry data missing or incomplete for sequence $SEQ.

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

Extract all three under DEST=$DEST. Expected post-extraction tree:
    $DEST/sequences/$SEQ/image_2/000000.png
    $DEST/sequences/$SEQ/image_2/000001.png
    ...
    $DEST/sequences/$SEQ/calib.txt
    $DEST/poses/$SEQ.txt
        (or equivalently: $DEST/sequences/$SEQ/poses.txt — loader accepts both)

Re-run this script after extraction to verify.
EOF
exit 1
