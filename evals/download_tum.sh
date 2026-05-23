#!/usr/bin/env bash
# Download a single TUM RGB-D Freiburg-1 sequence.
# Usage: bash evals/download_tum.sh desk /workspace/collab-splats/evals/data/tum
# Sequences: desk desk2 360 floor plant room rpy teddy xyz
set -euo pipefail

SEQ="${1:-desk}"
DEST="${2:-/workspace/collab-splats/evals/data/tum}"

VALID=(desk desk2 360 floor plant room rpy teddy xyz)
if [[ ! " ${VALID[*]} " =~ " ${SEQ} " ]]; then
    echo "Unknown sequence '$SEQ'. Choose from: ${VALID[*]}"
    exit 1
fi

NAME="rgbd_dataset_freiburg1_${SEQ}"
URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/${NAME}.tgz"

mkdir -p "$DEST"

# Skip if already extracted (handle both flat and nested layouts)
if [[ -d "$DEST/$NAME" && -f "$DEST/$NAME/groundtruth.txt" ]]; then
    echo "Already extracted: $DEST/$NAME"
    exit 0
fi
if [[ -d "$DEST/$SEQ" && -f "$DEST/$SEQ/groundtruth.txt" ]]; then
    echo "Already extracted: $DEST/$SEQ"
    exit 0
fi

TGZ="$DEST/${NAME}.tgz"
echo "Downloading $SEQ -> $TGZ"
wget -c "$URL" -O "$TGZ"
echo "Extracting..."
tar -xzf "$TGZ" -C "$DEST"
echo "Done. Data at $DEST/$NAME/"
echo ""
echo "Expected structure:"
echo "  $DEST/$NAME/rgb/*.png"
echo "  $DEST/$NAME/rgb.txt"
echo "  $DEST/$NAME/depth.txt"
echo "  $DEST/$NAME/groundtruth.txt"
