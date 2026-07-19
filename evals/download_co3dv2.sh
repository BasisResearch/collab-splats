#!/usr/bin/env bash
# Download a single CO3Dv2 category for evaluation.
#
# Usage:
#   bash evals/download_co3dv2.sh <category> [output_dir]
#
# <category>: one of apple, ball, banana, bench, book, bottle, bowl, broccoli, car, chair
# [output_dir]: destination directory (default: ./evals/data/co3dv2/)

set -euo pipefail

CATEGORY="${1:-}"
OUTPUT_DIR="${2:-./evals/data/co3dv2}"

if [ -z "$CATEGORY" ]; then
    echo "Usage: bash evals/download_co3dv2.sh <category> [output_dir]"
    echo "Categories: apple ball banana bench book bottle bowl broccoli car chair"
    exit 1
fi

STANDARD_CATEGORIES="apple ball banana bench book bottle bowl broccoli car chair"
if ! echo "$STANDARD_CATEGORIES" | grep -qw "$CATEGORY"; then
    echo "Warning: '$CATEGORY' is not in the standard 10-category eval set."
    echo "Standard categories: $STANDARD_CATEGORIES"
fi

echo "=== Downloading CO3Dv2 category: $CATEGORY → $OUTPUT_DIR ==="
mkdir -p "$OUTPUT_DIR"

PYTHON="${PYTHON:-/opt/venv/reconstruction/bin/python}"
if ! "$PYTHON" -c "import co3d" 2>/dev/null; then
    echo "Error: 'co3d' package not found. Install with:"
    echo "  $PYTHON -m pip install co3d"
    echo "  OR: git clone https://github.com/facebookresearch/co3d && pip install -e co3d/"
    exit 1
fi

CO3D_SCRIPT="$("$PYTHON" -c 'import co3d, os; print(os.path.join(os.path.dirname(co3d.__file__), "download_dataset.py"))')"

"$PYTHON" "$CO3D_SCRIPT" \
    --download_folder "$OUTPUT_DIR" \
    --download_categories "$CATEGORY" \
    --single_sequence_subset

echo "Download complete: ${OUTPUT_DIR}/${CATEGORY}/"

echo "=== Done. Test with: ==="
echo "python evals/eval.py --dataset co3dv2 \\"
echo "  --seq_dir ${OUTPUT_DIR}/${CATEGORY}/<sequence_name> \\"
echo "  --output_dir ./eval_results/co3dv2_${CATEGORY} \\"
echo "  --conditions baseline ba ba_hightrack"
