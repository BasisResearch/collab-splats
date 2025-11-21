# Clustering Script Usage Guide

## Quick Start

### Basic Usage (Recommended)

```bash
cd /workspace/collab-splats/docs/splats

# Semantic-first clustering with quality evaluation
python cluster_mesh.py --dataset rats_001 --n-clusters 11 \
    --semantic-weight 0.85 \
    --use-pca \
    --evaluate \
    --visualize
```

### Compare Different Semantic Weights

```bash
# Automatically tests weights: 0.70, 0.80, 0.85, 0.90, 0.95
python cluster_mesh.py --dataset rats_001 --n-clusters 11 \
    --use-pca \
    --compare-weights \
    --evaluate \
    --visualize
```

### GPU-Accelerated (if RAPIDS cuML available)

```bash
python cluster_mesh.py --dataset rats_001 --n-clusters 11 \
    --semantic-weight 0.85 \
    --use-pca \
    --use-gpu \
    --evaluate \
    --visualize
```

### Auto K-Selection (Elbow Method)

```bash
python cluster_mesh.py --dataset rats_001 --n-clusters auto \
    --k-range 5 25 \
    --semantic-weight 0.85 \
    --use-pca \
    --evaluate \
    --visualize
```

### HDBSCAN (Automatic Cluster Detection)

```bash
python cluster_mesh.py --dataset rats_001 --algorithm hdbscan \
    --min-cluster-size 500 \
    --use-pca \
    --evaluate \
    --visualize
```

---

## Key Parameters

### Semantic Weighting (Most Important!)

**`--semantic-weight <float>`** (default: 0.85)
- Range: 0.0 to 1.0
- Higher values = more semantic coherence, less spatial influence
- **Recommended values:**
  - 0.85-0.95: Strong semantic clustering (clusters by visual appearance)
  - 0.70-0.80: Balanced semantic + spatial
  - 0.50-0.65: More spatial influence (not recommended for semantic segmentation)

**Example:**
```bash
# High semantic weight - clusters walls, floors, objects separately
python cluster_mesh.py --dataset rats_001 --n-clusters 11 --semantic-weight 0.90
```

### Dimensionality Reduction

**`--use-pca`** - Apply PCA dimensionality reduction (recommended!)
**`--pca-variance <float>`** (default: 0.95) - Variance to preserve

**Benefits:**
- Faster clustering (768D → ~50-100D)
- Better cluster quality
- Reduced memory usage

**Example:**
```bash
# Use PCA to preserve 95% variance
python cluster_mesh.py --dataset rats_001 --n-clusters 11 --use-pca --pca-variance 0.95
```

### Quality Evaluation

**`--evaluate`** - Compute cluster quality metrics
**`--eval-sample-size <int>`** (default: 10000) - Sample size for metrics

**Metrics computed:**
- Silhouette Score (higher is better, range: -1 to 1)
- Davies-Bouldin Index (lower is better)
- Calinski-Harabasz Score (higher is better)

**Example:**
```bash
# Evaluate clustering quality
python cluster_mesh.py --dataset rats_001 --n-clusters 11 --evaluate
```

### Weight Comparison

**`--compare-weights`** - Compare multiple semantic weights automatically

Tests 5 different weights (0.70, 0.80, 0.85, 0.90, 0.95) and shows which performs best.

**Example:**
```bash
# Find optimal semantic weight objectively
python cluster_mesh.py --dataset rats_001 --n-clusters 11 \
    --use-pca --compare-weights --evaluate
```

### Post-Processing (Optional)

**`--smooth`** - Apply lightweight spatial smoothing (1 iteration)
**`--smooth-k <int>`** (default: 20) - Number of neighbors
**`--smooth-iterations <int>`** (default: 1) - Number of iterations

**Note:** Spatial smoothing can reduce semantic coherence. Use sparingly.

**Example:**
```bash
# Apply minimal spatial smoothing
python cluster_mesh.py --dataset rats_001 --n-clusters 11 --smooth --smooth-iterations 1
```

---

## Complete Examples

### Example 1: Quick Clustering with Evaluation

```bash
python cluster_mesh.py \
    --dataset rats_001 \
    --n-clusters 11 \
    --semantic-weight 0.85 \
    --use-pca \
    --evaluate \
    --visualize
```

**Output:**
```
======================================================================
CLUSTERING PIPELINE
======================================================================

--- STEP 1: DIMENSIONALITY REDUCTION ---
Applying PCA dimensionality reduction...
  Input shape: (53842, 768)
  Output shape: (53842, 87)
  Explained variance: 0.950
  Number of components: 87

--- STEP 2: CLUSTERING ---

Semantic-Spatial Clustering (semantic_weight=0.85)
  Semantic features: (53842, 87)
  Spatial positions: (53842, 3)
  Combined features: (53842, 90)

Cluster distribution:
  Cluster 0: 4,234 samples (7.9%)
  Cluster 1: 6,543 samples (12.2%)
  ...

--- EVALUATION: CLUSTER QUALITY METRICS ---
Silhouette Score: 0.234 (higher is better, [-1, 1])
Davies-Bouldin Index: 1.45 (lower is better)
Calinski-Harabasz Score: 1234.5 (higher is better)
```

### Example 2: Find Optimal Semantic Weight

```bash
python cluster_mesh.py \
    --dataset rats_001 \
    --n-clusters 11 \
    --use-pca \
    --compare-weights \
    --evaluate
```

**Output:**
```
--- COMPARISON: DIFFERENT SEMANTIC WEIGHTS ---

Trying semantic_weight=0.70...
Trying semantic_weight=0.80...
Trying semantic_weight=0.85...
Trying semantic_weight=0.90...
Trying semantic_weight=0.95...

======================================================================
CLUSTERING COMPARISON
======================================================================
Method                         Silhouette   Davies-Bouldin   Calinski-Harabasz
----------------------------------------------------------------------
Semantic_70                    0.198        1.62             987.3
Semantic_80                    0.221        1.53             1123.4
Semantic_85                    0.234        1.45             1234.5
Semantic_90                    0.245        1.39             1345.6
Semantic_95                    0.256        1.38             1456.2

✓ Best clustering: Semantic_95
```

### Example 3: GPU-Accelerated with HDBSCAN

```bash
python cluster_mesh.py \
    --dataset rats_001 \
    --algorithm hdbscan \
    --min-cluster-size 500 \
    --use-gpu \
    --use-pca \
    --evaluate \
    --visualize
```

**Output:**
```
✓ RAPIDS cuML available - GPU acceleration enabled

--- STEP 2: CLUSTERING ---

GPU HDBSCAN (min_cluster_size=500)...
Found 12 clusters, 342 noise points (0.6%)

Handling 342 noise points (0.6%)...
  All noise points reassigned

--- EVALUATION: CLUSTER QUALITY METRICS ---
Silhouette Score: 0.267 (higher is better, [-1, 1])
Davies-Bouldin Index: 1.32 (lower is better)
Calinski-Harabasz Score: 1567.8 (higher is better)
```

---

## Output Files

All results saved to `clustering_results_refactored/` (or custom `--output-dir`):

- `<dataset>_labels_k<N>.npy` - Cluster labels (NumPy array)
- `<dataset>_metadata_k<N>.pkl` - Metadata (cluster sizes, etc.)
- `<dataset>_metrics_k<N>.pkl` - Quality metrics (if `--evaluate`)
- `<dataset>_comparison_k<N>.pkl` - Weight comparison (if `--compare-weights`)
- `<dataset>_all_clusters_k<N>.png` - Visualization of all clusters (if `--visualize`)
- `<dataset>_cluster_panels_k<N>.png` - Individual cluster panels (if `--visualize`)

---

## Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'collab_splats.stage'`

**Fixed!** The script now uses direct imports. If you still see this, make sure you're using the updated version.

### Warning: `UMAP not available`

Optional dependency. Install with:
```bash
pip install umap-learn
```

### Warning: `SpLiCE not available`

Optional dependency for semantic interpretation. Install from:
```bash
git clone https://github.com/AI4LIFE-GROUP/SpLiCE.git
cd SpLiCE
pip install -e .
```

### Warning: `RAPIDS cuML not available`

GPU acceleration requires RAPIDS cuML. Install with:
```bash
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.10 python=3.11 cudatoolkit=11.8
```

Or run without `--use-gpu` flag (CPU fallback).

---

## Best Practices

### 1. Always Use PCA
```bash
--use-pca --pca-variance 0.95
```
Reduces dimensionality, improves clustering quality, and speeds up computation.

### 2. Always Evaluate
```bash
--evaluate
```
Get objective quality metrics instead of relying on visual inspection.

### 3. Compare Weights First
```bash
--compare-weights
```
Find the optimal semantic weight for your dataset objectively.

### 4. Use High Semantic Weight
```bash
--semantic-weight 0.85
```
Or higher (0.90, 0.95) for better semantic coherence.

### 5. Minimal Post-Processing
Avoid `--smooth` unless absolutely necessary. Trust the clustering algorithm.

### 6. Use GPU if Available
```bash
--use-gpu
```
10-100x faster for large datasets.

---

## Recommended Workflow

```bash
# Step 1: Find optimal K (if unknown)
python cluster_mesh.py --dataset rats_001 --n-clusters auto \
    --use-pca --evaluate

# Step 2: Find optimal semantic weight
python cluster_mesh.py --dataset rats_001 --n-clusters 11 \
    --use-pca --compare-weights --evaluate

# Step 3: Final clustering with best parameters
python cluster_mesh.py --dataset rats_001 --n-clusters 11 \
    --semantic-weight 0.95 \
    --use-pca \
    --evaluate \
    --visualize

# Step 4: (Optional) Use GPU for faster processing
python cluster_mesh.py --dataset rats_001 --n-clusters 11 \
    --semantic-weight 0.95 \
    --use-pca \
    --use-gpu \
    --visualize
```

---

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | (required) | Dataset name: `rats_001` or `birds_008` |
| `--algorithm` | `kmeans` | Clustering algorithm: `kmeans` or `hdbscan` |
| `--n-clusters` | `11` | Number of clusters or `auto` |
| `--k-range` | `5 25` | K range for auto selection |
| `--semantic-weight` | `0.85` | Semantic feature weight (0-1) |
| `--compare-weights` | `False` | Compare multiple weights |
| `--use-pca` | `False` | Apply PCA dimensionality reduction |
| `--pca-variance` | `0.95` | Variance to preserve with PCA |
| `--use-gpu` | `False` | Use GPU acceleration (cuML) |
| `--min-cluster-size` | `500` | Min cluster size (HDBSCAN) |
| `--smooth` | `False` | Apply spatial smoothing |
| `--smooth-k` | `20` | Neighbors for smoothing |
| `--smooth-iterations` | `1` | Smoothing iterations |
| `--evaluate` | `False` | Compute quality metrics |
| `--eval-sample-size` | `10000` | Sample size for metrics |
| `--visualize` | `False` | Create visualizations |
| `--output-dir` | `clustering_results_refactored/` | Output directory |

---

## FAQ

**Q: What's the difference between semantic_weight and the old spatial_weight?**

A: `semantic_weight` is clearer:
- `--semantic-weight 0.85` means 85% semantic, 15% spatial
- Old `--spatial-weight 0.15` was confusing (higher = less semantic)

**Q: Should I use spatial smoothing?**

A: Generally no. It can destroy semantic coherence. Only use if you absolutely need spatial continuity.

**Q: How do I know if my clustering is good?**

A: Use `--evaluate`:
- Silhouette > 0.2 is decent, > 0.3 is good
- Davies-Bouldin < 2.0 is good, < 1.5 is better
- Calinski-Harabasz: higher is better (no absolute scale)

**Q: What's the best semantic weight?**

A: Use `--compare-weights` to find out! Usually 0.85-0.95 works well for semantic segmentation.

**Q: Should I use PCA?**

A: Yes! Almost always. It improves quality and speed.

**Q: KMeans vs HDBSCAN?**

A:
- KMeans: Fast, known # clusters, good for structured scenes
- HDBSCAN: Automatic # clusters, handles noise, better for complex scenes

---

**For more details, see:**
- [CLUSTERING_IMPROVEMENTS.md](CLUSTERING_IMPROVEMENTS.md) - Detailed improvements guide
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Code refactoring details
- [clustering_utils.py](../../stage/clustering_utils.py) - Core utilities documentation
