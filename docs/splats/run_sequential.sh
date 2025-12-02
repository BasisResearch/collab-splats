#!/bin/bash

# Run mesh generation pipelines sequentially to reduce memory usage

datasets=(
    # "rats_date-07112024_video-C0119"  # Already completed
    "birds_date-02062024_video-C0043"
    "birds_date-06012024_video-GH010164"
    "birds_date-11052023_video-PXL_20231105_154956078"
)

cd /workspace/collab-splats

for dataset in "${datasets[@]}"; do
    echo "============================================"
    echo "Starting pipeline for: $dataset"
    echo "Time: $(date)"
    echo "============================================"

    python3 docs/splats/run_pipeline.py --dataset "$dataset" --overwrite

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed: $dataset"
    else
        echo "✗ Failed: $dataset (exit code: $exit_code)"
        echo "Stopping sequential processing due to error"
        exit 1
    fi

    echo ""
done

echo "============================================"
echo "All pipelines completed successfully!"
echo "Time: $(date)"
echo "============================================"
