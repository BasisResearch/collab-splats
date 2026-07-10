#!/usr/bin/env bash
# Download the LC-parity validation scenes (spec 2026-07-08-lc-parity-validation-design).
# Idempotent: skips anything already extracted. ~25 GB total — check disk first.
# 7-Scenes: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
# TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
#
# URLs and extraction mechanics cross-checked against the existing downloaders
# (evals/download_7scenes.py, evals/download_7scenes.sh, evals/download_tum.sh)
# for consistency with the proven-working scripts:
#   - 7-Scenes base URL is identical to download_7scenes.py's `_CDN` constant.
#   - TUM base URL + per-sequence .tgz path match download_tum.sh's `$URL`.
# Layout matches lc_parity_common.SCENES:
#   - 7-Scenes: evals/data/7scenes/<scene>/<scene>/seq-01 (Microsoft's outer zip
#     nests as <scene>/<scene>/seq-NN.zip; verified against the local chess/
#     download, which used the same download_7scenes.sh mechanics).
#   - TUM: evals/data/tum/rgbd_dataset_<name> (standard top-level dir name inside
#     every official TUM RGB-D .tgz archive).
set -euo pipefail
cd "$(dirname "$0")"

# 7-Scenes zips contain STORED entries with a size-mismatch quirk; unzip exits 1
# ("warnings, extraction OK"). Tolerate exit 1, fail on >=2 (real errors).
unzip_ok() { unzip -q -o "$@" || [ $? -eq 1 ]; }

SEVEN_SCENES_BASE="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
for scene in fire heads office pumpkin redkitchen stairs; do   # chess already local
    if [ -d "7scenes/${scene}/${scene}/seq-01" ]; then
        echo "[skip] 7scenes/${scene}"
        continue
    fi
    mkdir -p "7scenes/${scene}"
    echo "[get ] 7-Scenes ${scene}"
    wget -c "${SEVEN_SCENES_BASE}/${scene}.zip" -O "7scenes/${scene}.zip"
    unzip_ok "7scenes/${scene}.zip" -d "7scenes/${scene}"
    # scene zips contain per-seq zips; extract seq-01 only (parity uses seq-01)
    unzip_ok "7scenes/${scene}/${scene}/seq-01.zip" -d "7scenes/${scene}/${scene}"
done

TUM_BASE="https://cvg.cit.tum.de/rgbd/dataset"
declare -A TUM=(
    [freiburg1/rgbd_dataset_freiburg1_desk]=fr1_desk
    [freiburg1/rgbd_dataset_freiburg1_room]=fr1_room
    [freiburg2/rgbd_dataset_freiburg2_xyz]=fr2_xyz
    [freiburg3/rgbd_dataset_freiburg3_long_office_household]=fr3_office
)
mkdir -p tum
for path in "${!TUM[@]}"; do
    name="$(basename "${path}")"
    if [ -d "tum/${name}" ]; then
        echo "[skip] tum/${name}"
        continue
    fi
    echo "[get ] TUM ${TUM[$path]}"
    wget -c "${TUM_BASE}/${path}.tgz" -O "tum/${name}.tgz"
    tar -xzf "tum/${name}.tgz" -C tum/
done

echo "Done. Verify: ls 7scenes/*/*/seq-01 | head; ls tum/"
