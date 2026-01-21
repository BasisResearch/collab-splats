#!/bin/bash

# Batch script to run semantic dictionary segmentation on multiple datasets
# This script processes datasets that have already been run through run_pipeline or run_sequential

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_DIR="/workspace/collab-splats/docs/splats/configs"
SCRIPT_DIR="/workspace/collab-splats/stage"
DICT_FILE="${SCRIPT_DIR}/semantic_dictionary_example.json"

# Clustering parameters
THRESHOLD=0.95
RADIUS=0.02

# Array of datasets to process
DATASETS=(
    # "rats_date-07112024_video-C0119"
    # "birds_date-02062024_video-C0043"
    # "birds_date-06012024_video-GH010164"
    "birds_date-11052023_video-PXL_20231105_154956078"
)

# Log file
LOG_FILE="${SCRIPT_DIR}/semantic_batch_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Semantic Dictionary Batch Processing${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Processing ${#DATASETS[@]} datasets"
echo -e "Log file: ${LOG_FILE}"
echo -e "Similarity threshold: ${THRESHOLD}"
echo -e "Spatial radius: ${RADIUS}"
echo -e "${BLUE}================================================${NC}\n"

# Function to log messages
log() {
    echo -e "$1" | tee -a "${LOG_FILE}"
}

# Counter for successes and failures
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_DATASETS=()

# Process each dataset
for DATASET in "${DATASETS[@]}"; do
    log "${BLUE}================================================${NC}"
    log "${YELLOW}Processing: ${DATASET}${NC}"
    log "${BLUE}================================================${NC}"
    log "Config directory: ${CONFIG_DIR}"
    log "Started: $(date)"

    # Run semantic dictionary script
    if python3 "${SCRIPT_DIR}/run_semantic_dictionary.py" \
        --dataset "${DATASET}" \
        --config-dir "${CONFIG_DIR}" \
        --dict "${DICT_FILE}" \
        --threshold ${THRESHOLD} \
        --radius ${RADIUS} 2>&1 | tee -a "${LOG_FILE}"; then

        log "${GREEN}✓ Successfully processed ${DATASET}${NC}"
        log "Completed: $(date)\n"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log "${RED}✗ Failed to process ${DATASET}${NC}"
        log "Failed: $(date)\n"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        FAILED_DATASETS+=("${DATASET}")
    fi
done

# Print summary
log "${BLUE}================================================${NC}"
log "${BLUE}BATCH PROCESSING SUMMARY${NC}"
log "${BLUE}================================================${NC}"
log "Total datasets: ${#DATASETS[@]}"
log "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
log "${RED}Failed: ${FAILURE_COUNT}${NC}"

if [ ${FAILURE_COUNT} -gt 0 ]; then
    log "\n${RED}Failed datasets:${NC}"
    for FAILED in "${FAILED_DATASETS[@]}"; do
        log "  - ${FAILED}"
    done
fi

log "\n${BLUE}Log saved to: ${LOG_FILE}${NC}"
log "${BLUE}================================================${NC}"

# Exit with error if any dataset failed
if [ ${FAILURE_COUNT} -gt 0 ]; then
    exit 1
fi

exit 0
