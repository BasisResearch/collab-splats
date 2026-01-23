# Quick Start: Pushing Mesh Solutions to GCS

## Immediate Task: Push Two Specific Sessions

### Sessions to Push
1. `rats_date-07112024_video-C0119`
2. `birds_date-02062024_video-C0043`

### Quick Commands

**Option 1: Push both sessions at once**
```bash
cd /workspace/collab-splats
python scripts/push_mesh_to_gcloud.py --all
```

**Option 2: Push one at a time**
```bash
cd /workspace/collab-splats
python scripts/push_mesh_to_gcloud.py --session rats_date-07112024_video-C0119
python scripts/push_mesh_to_gcloud.py --session birds_date-02062024_video-C0043
```

**Force re-upload (if files already exist)**
```bash
python scripts/push_mesh_to_gcloud.py --all --force
```

### What Gets Uploaded

**Local paths:**
- `/workspace/fieldwork-data/rats/2024-07-11/environment/C0119/`
- `/workspace/fieldwork-data/birds/2024-02-06/environment/C0043/`

**GCS paths:**
- `fieldwork_processed/processed_splats/2024-07-11/`
- `fieldwork_processed/processed_splats/2024-02-06/`

**Contents:**
- `preproc/` - Preprocessed data (COLMAP, transforms, etc.)
- `rade-features/` - Trained model and mesh outputs
  - Timestamped run directories (e.g., `2025-12-16_050805/`)
  - `mesh/` - Mesh files (`.ply`, `.pt`, `.yaml`)
  - `query/` - Query results
  - Model checkpoints

**Excluded:**
- `images_4/`, `images_8/` - Downsampled images (to save space)

### Prerequisites

1. **Environment variables set up** (`.env` file):
   ```bash
   COLLAB_DATA_KEY=/workspace/api-keys/collab-data-463313-c340ad86b28e.json
   ```

2. **Dependencies installed**:
   ```bash
   pip install python-dotenv
   # collab-data package should already be installed
   ```

### Verify Upload

After uploading, you can verify using the [gcloud_data_interface notebook](../data/gcloud_data_interface.ipynb):

```python
from collab_data.gcs_utils import GCSClient
import os
from pathlib import Path

# Initialize client
project_key = Path(os.environ.get("COLLAB_DATA_KEY"))
project_id = "-".join(project_key.stem.split("-")[:-1])
gcs_client = GCSClient(project_id=project_id, credentials_path=project_key)

# Check what was uploaded
files = gcs_client.glob("fieldwork_processed/processed_splats/2024-02-06/**")
print(f"Found {len(files)} files for birds session")

files = gcs_client.glob("fieldwork_processed/processed_splats/2024-07-11/**")
print(f"Found {len(files)} files for rats session")
```

## Programmatic Usage

You can also push/pull data programmatically using the GCS helper functions:

**Push after pipeline:**
```python
from collab_splats.wrapper import Splatter
from collab_splats.utils import push_to_gcs

splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')
splatter.run_pipeline()
push_to_gcs(splatter)  # Auto-detects date and uploads
```

**Pull existing data:**
```python
from collab_splats.wrapper import Splatter
from collab_splats.utils import pull_from_gcs

splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')
pull_from_gcs(splatter, include=['mesh'])  # Only download mesh files
splatter.query_mesh(queries=['tree', 'feeder'])
```

**See [examples/gcs_sync_example.py](examples/gcs_sync_example.py) for more examples.**

## Troubleshooting

**Error: "COLLAB_DATA_KEY not found"**
- Ensure `.env` file exists and contains the key path
- Check that `load_dotenv()` is called before accessing the environment variable

**Error: "Local directory does not exist"**
- Verify the session naming matches the actual directory structure
- Check that the pipeline has completed and generated outputs

**Error: "Permission denied"**
- Verify GCS credentials file exists at the specified path
- Check that credentials have write access to the `fieldwork_processed` bucket

**Upload is slow**
- Large mesh files can take time (especially `.ply` files with features)
- Consider using `--skip-existing` to avoid re-uploading
- Use `include_patterns` to upload only specific file types

## Session Naming Convention

The script parses session names with this format:
```
{species}_date-{MMDDYYYY}_video-{VIDEO_ID}
```

Examples:
- `rats_date-07112024_video-C0119` → `/workspace/fieldwork-data/rats/2024-07-11/environment/C0119/`
- `birds_date-02062024_video-C0043` → `/workspace/fieldwork-data/birds/2024-02-06/environment/C0043/`

The GCS path is automatically generated:
```
fieldwork_processed/processed_splats/{YYYY-MM-DD}/
```
