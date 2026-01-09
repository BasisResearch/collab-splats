# Semantic Dictionary Segmentation Script

This script performs semantic segmentation using a dictionary of terms with positive queries. Each term's queries are contrasted against negative queries from other dictionary items to segment the environment into semantic compartments.

## Overview

The script takes a processed configuration file (from `run_pipeline` or `run_sequential`), runs semantic dictionary queries on the mesh, clusters the results, and generates visualizations.

## Usage

### Single Dataset

```bash
# Basic usage
python run_semantic_dictionary.py --dataset rats_date-07112024_video-C0119

# With custom parameters
python run_semantic_dictionary.py \
    --dataset birds_date-02062024_video-C0043 \
    --threshold 0.90 \
    --radius 0.03

# With custom dictionary
python run_semantic_dictionary.py \
    --dataset birds_date-02062024_video-C0043 \
    --dict my_custom_dictionary.json
```

### Batch Processing (Multiple Datasets)

Process all datasets defined in the batch script:

```bash
./run_semantic_batch.sh
```

This will process all datasets sequentially and generate a log file with results.

## Arguments

- `--config` (required): Path to the YAML config file
  - Example: `docs/splats/configs/birds_date-02062024_video-C0043.yaml`

- `--dict` (optional): Path to semantic dictionary JSON file
  - If not provided, uses default dictionary
  - See `semantic_dictionary_example.json` for format

- `--threshold` (optional): Similarity threshold for clustering (default: 0.95)
  - Range: 0.0 to 1.0
  - Higher values = stricter clustering

- `--radius` (optional): Spatial radius for clustering (default: 0.02)
  - Controls spatial proximity for clustering

## Semantic Dictionary Format

The semantic dictionary is a JSON file mapping category names to lists of query terms:

```json
{
  "tree": ["green", "leaves", "bark", "trunk"],
  "feeder": ["bird feeder", "container", "food"],
  "brush": ["leaves", "plants", "thicket", "bramble"],
  "gravel": ["gravel", "rock", "concrete"]
}
```

Each category will be queried using its positive terms, contrasted against the other categories as negative queries.

## Output Files

All output files are saved to the mesh directory within the processed dataset:

- `all_clusters.ply` - 3D mesh with all clusters colored by category
- `all_clusters_view.png` - Combined visualization of all clusters
- `clusters_grid.png` - Grid showing each cluster individually
- `cluster_labels.npy` - NumPy array of cluster labels per vertex
- `query-{category}.ply` - Individual query results for each category

### Example Output Location

For a dataset at `/workspace/fieldwork-data/birds/2024-02-06/environment/C0043`,
outputs will be saved to:
```
/workspace/fieldwork-data/birds/2024-02-06/environment/C0043/rade-features/{run_id}/mesh/
```

## Example

```bash
# Run semantic dictionary on a processed bird dataset
python run_semantic_dictionary.py \
    --config /workspace/collab-splats/docs/splats/configs/birds_date-02062024_video-C0043.yaml

# Run with custom dictionary and parameters
python run_semantic_dictionary.py \
    --config /workspace/collab-splats/docs/splats/configs/birds_date-02062024_video-C0043.yaml \
    --dict my_custom_dictionary.json \
    --threshold 0.92 \
    --radius 0.025
```

## Requirements

The script requires:
- A processed dataset (run through `run_pipeline` or `run_sequential`)
- The dataset must have:
  - Completed preprocessing (`transforms.json` exists)
  - Extracted features (RADE features)
  - Generated mesh

If these steps haven't been completed, the script will attempt to run them automatically.

## Workflow

1. **Load Configuration**: Loads the splatter from the YAML config file
2. **Ensure Processing**: Verifies preprocessing, features, and mesh are complete
3. **Query Categories**: Runs semantic queries for each category in the dictionary
4. **Cluster Results**: Clusters high-similarity regions for each category
5. **Create Visualizations**: Generates combined and individual cluster visualizations
6. **Save Outputs**: Saves all meshes and images to the mesh directory

## Tips

- **Adjust threshold**: If you get too many small clusters, increase `--threshold`
- **Adjust radius**: If clusters are too fragmented, increase `--radius`
- **Custom dictionaries**: Create domain-specific dictionaries for different environments
  - Urban scenes: buildings, roads, vegetation, sky
  - Indoor scenes: walls, floor, furniture, windows
  - Natural scenes: trees, ground, water, sky
