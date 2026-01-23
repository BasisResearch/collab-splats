# GCS Integration Proposal for Splatter Class

## Overview

This document proposes integrating Google Cloud Storage (GCS) push/pull functionality into the `Splatter` class to enable easy data synchronization based on configuration files.

## Current State

- **Manual uploads**: Using [gcloud_data_interface.ipynb](../data/gcloud_data_interface.ipynb) for one-off uploads
- **Separate workflow**: Data management is disconnected from the splatting pipeline
- **No automation**: No way to automatically sync results after processing

## Proposed Integration

### Option 1: Add GCS Methods to Splatter Class (Recommended)

**Pros:**
- Keeps everything in one place
- Natural workflow: `splatter.run_pipeline()` → `splatter.push_to_gcs()`
- Can use config file to determine GCS paths
- Easy to add to existing scripts

**Cons:**
- Adds GCS dependency to core class
- Slightly increases class complexity

**Implementation:**

```python
# In collab_splats/wrapper/splatter.py

class Splatter:
    def __init__(self, config: SplatterConfig):
        # ... existing code ...
        self._gcs_client: Optional[GCSClient] = None

    def _init_gcs_client(self) -> GCSClient:
        """Initialize GCS client from environment variables."""
        if self._gcs_client is None:
            load_dotenv()
            project_key = Path(os.environ.get("COLLAB_DATA_KEY"))
            project_id = "-".join(project_key.stem.split("-")[:-1])

            self._gcs_client = GCSClient(
                project_id=project_id,
                credentials_path=expand_path(project_key.as_posix(), get_project_root()),
            )
        return self._gcs_client

    def push_to_gcs(
        self,
        bucket: str = "fieldwork_processed",
        gcs_prefix: Optional[str] = None,
        skip_existing: bool = True,
        exclude_patterns: Optional[List[str]] = None
    ) -> int:
        """
        Push output directory to Google Cloud Storage.

        Args:
            bucket: GCS bucket name (default: 'fieldwork_processed')
            gcs_prefix: GCS prefix path. If None, auto-generated from config
            skip_existing: Skip files that already exist in GCS
            exclude_patterns: Patterns to exclude (e.g., ['images_4', 'images_8'])

        Returns:
            Number of files uploaded

        Example:
            >>> splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')
            >>> splatter.run_pipeline()
            >>> splatter.push_to_gcs()  # Auto-generates path from config
        """
        output_path = self.config["output_path"]

        if gcs_prefix is None:
            gcs_prefix = self._generate_gcs_prefix()

        gcs_client = self._init_gcs_client()

        # Upload directory
        uploaded_count = 0
        existing_files = set()

        if skip_existing:
            existing_files = set(gcs_client.glob(f"{gcs_prefix}/**"))

        for root, _, files in os.walk(output_path):
            for file in files:
                local_path = Path(root) / file
                relative_path = local_path.relative_to(output_path)

                # Check exclude patterns
                if exclude_patterns:
                    if any(pattern in str(relative_path) for pattern in exclude_patterns):
                        continue

                gcs_path = f"{gcs_prefix}/{relative_path.as_posix()}"

                if skip_existing and gcs_path in existing_files:
                    continue

                gcs_client.upload_file(str(local_path), gcs_path)
                uploaded_count += 1

        return uploaded_count

    def pull_from_gcs(
        self,
        gcs_prefix: Optional[str] = None,
        overwrite: bool = False,
        include_patterns: Optional[List[str]] = None
    ) -> int:
        """
        Pull data from Google Cloud Storage to local output directory.

        Args:
            gcs_prefix: GCS prefix path. If None, auto-generated from config
            overwrite: Overwrite local files if they exist
            include_patterns: Only download files matching these patterns

        Returns:
            Number of files downloaded

        Example:
            >>> splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')
            >>> splatter.pull_from_gcs(include_patterns=['mesh'])  # Only download mesh files
        """
        output_path = self.config["output_path"]

        if gcs_prefix is None:
            gcs_prefix = self._generate_gcs_prefix()

        gcs_client = self._init_gcs_client()

        # Download files
        downloaded_count = 0
        gcs_files = gcs_client.glob(f"{gcs_prefix}/**")

        for gcs_path in gcs_files:
            relative_path = gcs_path.replace(f"{gcs_prefix}/", "")

            # Check include patterns
            if include_patterns:
                if not any(pattern in relative_path for pattern in include_patterns):
                    continue

            local_path = output_path / relative_path

            if local_path.exists() and not overwrite:
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            gcs_client.download_file(gcs_path, str(local_path))
            downloaded_count += 1

        return downloaded_count

    def _generate_gcs_prefix(self) -> str:
        """
        Generate GCS prefix from config file path.

        Example config: 'birds_date-02062024_video-C0043.yaml'
        Generated prefix: 'fieldwork_processed/processed_splats/2024-02-06'
        """
        # Extract session info from file_path or config name
        file_path = Path(self.config["file_path"])
        video_name = file_path.stem  # e.g., 'C0043'

        # Parse date from parent directories or config
        # This assumes structure: /workspace/fieldwork-data/{species}/{date}/...
        date_parts = file_path.parts

        # Find date in format YYYY-MM-DD
        date_str = None
        for part in date_parts:
            if len(part) == 10 and part.count('-') == 2:
                date_str = part  # Keep hyphens in date
                break

        if date_str is None:
            raise ValueError("Could not determine date from file path")

        # Generate GCS prefix
        gcs_prefix = f"fieldwork_processed/processed_splats/{date_str}"

        return gcs_prefix
```

### Option 2: Create Separate GCSManager Class

**Pros:**
- Separation of concerns
- Can be used independently of Splatter
- Easier to test in isolation

**Cons:**
- Extra class to maintain
- Less integrated workflow

**Implementation:**

```python
# In collab_splats/data/gcs_manager.py

from pathlib import Path
from typing import Optional, List
import os
from dotenv import load_dotenv

from collab_data.file_utils import expand_path, get_project_root
from collab_data.gcs_utils import GCSClient


class GCSManager:
    """Manages Google Cloud Storage operations for Splatter outputs."""

    def __init__(self, credentials_env_var: str = "COLLAB_DATA_KEY"):
        """Initialize GCS manager."""
        load_dotenv()
        project_key = Path(os.environ.get(credentials_env_var))
        project_id = "-".join(project_key.stem.split("-")[:-1])

        self.client = GCSClient(
            project_id=project_id,
            credentials_path=expand_path(project_key.as_posix(), get_project_root()),
        )

    def push_splatter_output(
        self,
        splatter_config: dict,
        bucket: str = "fieldwork_processed",
        **kwargs
    ):
        """Push Splatter output based on config."""
        # Implementation similar to Option 1
        pass

    def pull_splatter_output(
        self,
        splatter_config: dict,
        **kwargs
    ):
        """Pull Splatter output based on config."""
        pass
```

### Option 3: Configuration-Driven Approach (Hybrid)

**Pros:**
- Declarative: specify GCS settings in config file
- Most flexible
- Can enable auto-sync after pipeline completion

**Cons:**
- Requires config schema changes
- More initial setup

**Implementation:**

Add GCS section to config files:

```yaml
# In docs/splats/configs/datasets/birds_date-02062024_video-C0043.yaml

file_path: /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4
frame_proportion: 0.25

# GCS sync configuration
gcs:
  enabled: true
  bucket: fieldwork_processed
  prefix: auto  # or specify manually: "2024_02_06-session_0001/environment/C0043"
  auto_push: false  # Auto-push after pipeline completion
  auto_pull: false  # Auto-pull before pipeline starts
  exclude_patterns:
    - images_4
    - images_8
  include_patterns:  # If specified, only sync matching files
    - mesh
    - preproc/transforms.json
```

Then modify `Splatter.run_pipeline()`:

```python
def run_pipeline(self, **kwargs):
    """Run the full pipeline with optional GCS sync."""

    # Auto-pull if configured
    if self.config.get("gcs", {}).get("auto_pull", False):
        self.pull_from_gcs()

    # Run pipeline steps
    self.preprocess(**kwargs)
    self.extract_features(**kwargs)
    self.mesh(**kwargs)

    # Auto-push if configured
    if self.config.get("gcs", {}).get("auto_push", False):
        self.push_to_gcs()
```

## Recommended Approach

**Hybrid of Options 1 & 3:**

1. Add `push_to_gcs()` and `pull_from_gcs()` methods to `Splatter` class
2. Support optional GCS configuration in YAML files
3. Keep it simple initially - don't require GCS config for basic usage
4. Add convenience script for batch operations

### Implementation Steps

1. **Add GCS methods to Splatter class** ([splatter.py](../collab_splats/wrapper/splatter.py))
   - `push_to_gcs()`
   - `pull_from_gcs()`
   - `_generate_gcs_prefix()` helper

2. **Update SplatterConfig TypedDict** to support optional GCS settings

3. **Create convenience scripts**:
   - `scripts/push_mesh_to_gcloud.py` (already created)
   - `scripts/sync_batch_sessions.py` (for bulk operations)

4. **Add examples to documentation**:
   - Update [run_pipeline.py](../docs/splats/run_pipeline.py)
   - Add notebook example

### Usage Examples

**Simple usage (manual sync):**
```python
from collab_splats.wrapper import Splatter

# Run pipeline
splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')
splatter.run_pipeline()

# Push results to GCS
splatter.push_to_gcs()
```

**Pull existing results:**
```python
# Download mesh files from GCS
splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')
splatter.pull_from_gcs(include_patterns=['mesh'])

# Load and query the mesh
splatter.query_mesh(queries=['tree', 'feeder'], output_fn='query_results.ply')
```

**Batch script:**
```bash
# Push specific sessions
python scripts/push_mesh_to_gcloud.py --session birds_date-02062024_video-C0043
python scripts/push_mesh_to_gcloud.py --session rats_date-07112024_video-C0119

# Push all pending sessions
python scripts/push_mesh_to_gcloud.py --all
```

**Configuration-driven (advanced):**
```yaml
# birds_date-02062024_video-C0043.yaml
file_path: /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4
frame_proportion: 0.25

gcs:
  enabled: true
  auto_push: true  # Push after pipeline completion
  exclude_patterns: [images_4, images_8]
```

```python
splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')
splatter.run_pipeline()  # Will auto-push if configured
```

## Migration Path

1. **Phase 1** (Immediate): Use standalone script for urgent uploads
2. **Phase 2** (This week): Integrate basic methods into Splatter class
3. **Phase 3** (Optional): Add configuration-driven auto-sync
4. **Phase 4** (Future): Add sync status tracking, conflict resolution, etc.

## Testing Strategy

1. Create test config with small dataset
2. Test push/pull roundtrip
3. Verify skip_existing works correctly
4. Test include/exclude patterns
5. Integration test with full pipeline

## Dependencies

- `collab-data` package (already installed)
- `.env` file with `COLLAB_DATA_KEY` variable
- GCS credentials JSON file

## Questions to Resolve

1. **Session numbering**: How to handle `session_0001` suffix? Auto-detect or configure?
2. **Conflict resolution**: What happens if GCS and local differ?
3. **Partial syncs**: Should we support syncing only mesh/ or only preproc/?
4. **Progress tracking**: Add progress bars for large uploads?
5. **Versioning**: How to handle multiple mesh runs for same session?

## Next Steps

1. Review this proposal
2. Decide on preferred approach (recommend Option 1 + 3 hybrid)
3. Implement `push_to_gcs()` and `pull_from_gcs()` methods
4. Test with the two sessions mentioned
5. Document usage patterns
