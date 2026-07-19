#!/usr/bin/env bash
# Download a single 7-Scenes sequence and extract recursively.
#
# Usage: bash evals/download_7scenes.sh <scene> [dest]
#   bash evals/download_7scenes.sh chess  ./evals/data/7scenes
#   bash evals/download_7scenes.sh fire   ./evals/data/7scenes
#   bash evals/download_7scenes.sh office ./evals/data/7scenes
#
# All 7-Scenes scenes are accepted: chess fire heads office pumpkin redkitchen stairs.
# Validated end-to-end against `evals/eval.py` (VGGT-SLAM-comparable evals): chess, fire, office.
# Other scenes share the same layout and should work but have not been benchmarked.
#
# Layout produced (matches `evals/datasets.py:_load_7scenes`):
#   <dest>/<scene>/seq-01/frame-000000.color.png
#   <dest>/<scene>/seq-01/frame-000000.pose.txt
#   ... (additional seq-NN directories for some scenes)
#
# The outer zip nests as <scene>/<scene>/seq-NN.zip and each seq-NN.zip must also
# be extracted, so this script flattens the outer wrapper and unzips every inner
# seq-NN.zip in place. Without that recursion the loader cannot find frame files.
set -euo pipefail

SCENE="${1:-chess}"
DEST="${2:-./evals/data/7scenes}"

declare -A URLS=(
    [chess]="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip"
    [fire]="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip"
    [heads]="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip"
    [office]="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
    [pumpkin]="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip"
    [redkitchen]="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip"
    [stairs]="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip"
)

if [[ -z "${URLS[$SCENE]+x}" ]]; then
    echo "Unknown scene '$SCENE'. Choose from: ${!URLS[@]}"
    exit 1
fi

mkdir -p "$DEST"
ZIP="$DEST/$SCENE.zip"
echo "Downloading $SCENE -> $ZIP"
wget -c "${URLS[$SCENE]}" -O "$ZIP"

# Stage extraction in a temp dir so we can flatten the nested wrapper that
# Microsoft's archive uses (outer: <scene>/<scene>/seq-NN.zip).
STAGE="$DEST/.${SCENE}_stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
echo "Extracting outer zip..."
unzip -q "$ZIP" -d "$STAGE"

# Locate the directory that actually contains seq-NN.zip files.
INNER="$(find "$STAGE" -maxdepth 4 -type f -name 'seq-*.zip' -printf '%h\n' | head -n 1 || true)"
if [[ -z "$INNER" ]]; then
    echo "ERROR: no seq-*.zip files found inside $ZIP" >&2
    exit 2
fi

# Move/merge inner contents to <dest>/<scene>/.
TARGET="$DEST/$SCENE"
mkdir -p "$TARGET"
shopt -s dotglob
mv "$INNER"/* "$TARGET"/
shopt -u dotglob
rm -rf "$STAGE"

echo "Extracting inner seq-*.zip files..."
for inner_zip in "$TARGET"/seq-*.zip; do
    [[ -e "$inner_zip" ]] || continue
    seq_name="$(basename "$inner_zip" .zip)"
    unzip -q -o "$inner_zip" -d "$TARGET/$seq_name"
    rm -f "$inner_zip"
done

echo "Done. Data at $TARGET/"
echo ""
echo "Expected structure:"
echo "  $TARGET/seq-01/frame-000000.color.png"
echo "  $TARGET/seq-01/frame-000000.pose.txt"
