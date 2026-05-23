#!/usr/bin/env bash
# Waymo Open Dataset download guide.
#
# Waymo data is license-gated: you must register and accept the dataset terms
# at https://waymo.com/open/ before downloading. This script does NOT
# attempt the download for you. Instead, it prints the steps and verifies
# that a destination layout matches what `evals/datasets.py:_load_waymo`
# expects after extraction by `evals/runners/extract_waymo.py`.
#
# Usage:
#   bash evals/download_waymo.sh <segment_id> <dest_dir>
#
# After download + extraction, the layout MUST be:
#   <dest_dir>/<segment_id>/images/000000.png ... 000NNN.png
#   <dest_dir>/<segment_id>/groundtruth.txt    (TUM format c2w)

set -e

SEG="${1:?usage: bash evals/download_waymo.sh <segment_id> <dest_dir>}"
DEST="${2:-/workspace/collab-splats/evals/data/waymo}"

mkdir -p "$DEST"
TARGET="$DEST/$SEG"

if [[ -f "$TARGET/groundtruth.txt" && -d "$TARGET/images" ]]; then
    echo "Waymo segment '$SEG' already extracted at $TARGET"
    exit 0
fi

cat <<EOF
Waymo Open Dataset is license-gated. To prepare segment '$SEG':

  1. Register at https://waymo.com/open/ and accept the dataset terms.
  2. Download the v1.4.1 segment .tfrecord, e.g. via gsutil:

       gsutil cp gs://waymo_open_dataset_v_1_4_1/individual_files/training/${SEG}.tfrecord \\
                "$DEST/${SEG}.tfrecord"

  3. Extract with the sidecar script (in a python env that has
     waymo-open-dataset installed — TensorFlow conflicts with our torch
     stack, keep it isolated):

       python evals/runners/extract_waymo.py \\
           --tfrecord "$DEST/${SEG}.tfrecord" \\
           --output   "$TARGET" \\
           --camera   FRONT

  4. The extracted layout is what _load_waymo consumes.

Re-run this script after extraction; it will exit 0 once the expected
layout is present at $TARGET.
EOF
exit 1
