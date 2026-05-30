#!/usr/bin/env bash
# eval_suite.sh — orchestrate 6-condition 7-Scenes eval
# Usage: bash evals/eval_suite.sh [--seq seq-01] [--submap_size 20] [--skip_download]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/venv/reconstruction/bin/python}"
EVALS_DIR="${REPO_ROOT}/evals"
DATA_DIR="${REPO_ROOT}/data/7scenes"
RESULTS_BASE="${REPO_ROOT}/evals/results"
SEQ="seq-01"
SUBMAP_SIZE=20
SKIP_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seq)           SEQ="$2"; shift 2 ;;
        --submap_size)   SUBMAP_SIZE="$2"; shift 2 ;;
        --skip_download) SKIP_DOWNLOAD=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCENES=("chess" "fire" "office")

# Step 1: Download missing sequences
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    echo "=== Downloading missing 7-Scenes sequences ==="
    "$PYTHON" "${EVALS_DIR}/download_7scenes.py" --scenes fire office --seq "$SEQ"
fi

for SCENE in "${SCENES[@]}"; do
    SEQ_DIR="${DATA_DIR}/${SCENE}/${SEQ}"
    SEQ_TAG="${SCENE}_$(echo "${SEQ}" | tr -d '-')"   # chess_seq01
    RESULTS_DIR="${RESULTS_BASE}/${SEQ_TAG}"
    mkdir -p "${RESULTS_DIR}"

    echo ""
    echo "===== Scene: ${SCENE}/${SEQ} → ${RESULTS_DIR} ====="

    # --- VGGT-Omega conditions ---
    for COND in baseline ba lc incremental_ba-3; do
        PREFIX="omega"
        TUM="${RESULTS_DIR}/${PREFIX}_${COND}.tum"
        if [[ -f "$TUM" ]]; then
            echo "  Skipping ${PREFIX}_${COND} (already exists)"
            continue
        fi
        echo "  Running ${PREFIX}_${COND} ..."
        "$PYTHON" "${EVALS_DIR}/eval_gt.py" \
            --dataset 7scenes \
            --seq_dir "${SEQ_DIR}" \
            --output_dir "${RESULTS_DIR}" \
            --backbone vggt_omega \
            --submap_size "${SUBMAP_SIZE}" \
            --conditions "${COND}"
    done

    # --- VGGT-X conditions ---
    for COND in baseline lc; do
        PREFIX="vggtx"
        TUM="${RESULTS_DIR}/${PREFIX}_${COND}.tum"
        if [[ -f "$TUM" ]]; then
            echo "  Skipping ${PREFIX}_${COND} (already exists)"
            continue
        fi
        echo "  Running ${PREFIX}_${COND} ..."
        "$PYTHON" "${EVALS_DIR}/eval_gt.py" \
            --dataset 7scenes \
            --seq_dir "${SEQ_DIR}" \
            --output_dir "${RESULTS_DIR}" \
            --backbone vggtx \
            --submap_size "${SUBMAP_SIZE}" \
            --conditions "${COND}"
    done

    # --- VGGT-SLAM ---
    SLAM_TUM="${RESULTS_DIR}/vggt_slam.tum"
    if [[ -f "$SLAM_TUM" ]]; then
        echo "  Skipping vggt_slam (already exists)"
    else
        echo "  Running vggt_slam ..."
        "$PYTHON" "${EVALS_DIR}/runners/run_vggt_slam.py" \
            --image_dir "${SEQ_DIR}" \
            --output "${SLAM_TUM}" \
            --submap_size "${SUBMAP_SIZE}"
    fi

    # --- eval_compare ---
    echo "  Running eval_compare ..."
    "$PYTHON" "${EVALS_DIR}/eval_compare.py" \
        --results-dir "${RESULTS_DIR}"

    echo "  metrics.json → ${RESULTS_DIR}/metrics.json"
done

echo ""
echo "=== All scenes complete ==="
