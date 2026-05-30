#!/usr/bin/env bash
# Run all non-splats tutorial notebooks in dependency order.
# Outputs go to /tmp/nb_out/. Stops on first failure.
set -euo pipefail

export PYVISTA_OFF_SCREEN=1        # config picks this up → "static" backend (no X server needed)
export COLLAB_SPLATS_CACHE=/workspace/outputs  # point get_cache_dir at existing data

PYTHON=/opt/conda/envs/reconstruction/bin/python
NB_DIR=/workspace/collab-splats/docs/source/tutorials
OUT_DIR=/tmp/nb_out
mkdir -p "$OUT_DIR"

run_nb() {
    local nb="$1"
    local name
    name=$(basename "$nb" .ipynb)
    echo ""
    echo "============================================================"
    echo "RUNNING: $nb"
    echo "============================================================"
    $PYTHON -m jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=900 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output "$OUT_DIR/${name}_out.ipynb" \
        "$nb" 2>&1
    echo "  DONE: $name"
}

# 01 — Preprocessing (generates images + frame_scores.json)
run_nb "$NB_DIR/01_preprocessing/keyframe_extraction.ipynb"

# 02 — Pointcloud (feedforward first; others depend on its zarr output)
run_nb "$NB_DIR/02_pointcloud/feedforward_methods.ipynb"
run_nb "$NB_DIR/02_pointcloud/bundle_adjustment.ipynb"
run_nb "$NB_DIR/02_pointcloud/feedforward_mesh.ipynb"
run_nb "$NB_DIR/02_pointcloud/slam_loop_closure.ipynb"
run_nb "$NB_DIR/02_pointcloud/colmap_sfm.ipynb"

# 02 — Stage notebooks
run_nb "$NB_DIR/02_pointcloud/stage/colmap_sfm.ipynb"
run_nb "$NB_DIR/02_pointcloud/stage/slam_loop_closure.ipynb"

# 04 — Semantics (image-only, no GPU required for most)
run_nb "$NB_DIR/04_semantics/feature_extraction.ipynb"
run_nb "$NB_DIR/04_semantics/maskclip_vs_talk2dino.ipynb"
run_nb "$NB_DIR/04_semantics/segmentation.ipynb"

# 05 — Lifting (depends on feedforward_methods + semantics)
run_nb "$NB_DIR/05_lifting/semantic_lifting.ipynb"

# 06 — Mesh
run_nb "$NB_DIR/06_mesh/create_mesh.ipynb"
run_nb "$NB_DIR/06_mesh/feedforward_mesh.ipynb"
run_nb "$NB_DIR/06_mesh/stage/feedforward_mesh.ipynb"
run_nb "$NB_DIR/06_mesh/stage/mesh_method_comparison.ipynb"

# 07 — Localization
run_nb "$NB_DIR/07_localization/localization.ipynb"

# Evals
run_nb "$NB_DIR/evals/ground_truth_evals.ipynb"

echo ""
echo "============================================================"
echo "ALL NOTEBOOKS COMPLETED SUCCESSFULLY"
echo "============================================================"
