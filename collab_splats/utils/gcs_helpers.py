"""Simple helpers to sync Splatter data with GCS."""

import os
from pathlib import Path


def push_to_gcs(splatter, skip_existing=True, exclude=None):
    """
    Upload Splatter output to GCS.

    Requires: pip install collab-data

    Args:
        splatter: Splatter instance
        skip_existing: Skip files already in GCS
        exclude: List of patterns to exclude (default: ['images_4', 'images_8'])

    Returns:
        int: Number of files uploaded

    Example:
        >>> from collab_splats.wrapper import Splatter
        >>> from collab_splats.utils.gcs_helpers import push_to_gcs
        >>> splatter = Splatter.from_config_file('config.yaml')
        >>> splatter.run_pipeline()
        >>> push_to_gcs(splatter)
    """
    from collab_data.gcs_utils import GCSClient
    from collab_data.file_utils import expand_path, get_project_root
    from dotenv import load_dotenv

    load_dotenv()

    # Init client
    key = Path(os.environ["COLLAB_DATA_KEY"])
    project_id = "-".join(key.stem.split("-")[:-1])
    client = GCSClient(
        project_id=project_id,
        credentials_path=expand_path(key.as_posix(), get_project_root())
    )

    # Get paths
    output_path = Path(splatter.config["output_path"])
    date = next(p for p in output_path.parts if len(p) == 10 and p.count("-") == 2)
    gcs_prefix = f"fieldwork_processed/processed_splats/{date}"

    print(f"Uploading to: {gcs_prefix}")

    # Default excludes
    if exclude is None:
        exclude = ['images_4', 'images_8']

    # Upload
    existing = set(client.glob(f"{gcs_prefix}/**")) if skip_existing else set()
    if skip_existing:
        print(f"Found {len(existing)} existing files in GCS")

    count = 0

    for root, _, files in os.walk(output_path):
        for file in files:
            local = Path(root) / file
            rel = local.relative_to(output_path)

            # Skip excluded patterns
            if any(ex in str(rel) for ex in exclude):
                continue

            gcs_path = f"{gcs_prefix}/{rel.as_posix()}"

            if gcs_path not in existing:
                print(f"📤 {rel}")
                client.upload_file(str(local), gcs_path)
                count += 1
            else:
                print(f"⏭️  {rel}")

    print(f"\n✅ Uploaded {count} files")
    return count


def pull_from_gcs(splatter, include=None, overwrite=False):
    """
    Download data from GCS to Splatter output path.

    Requires: pip install collab-data

    Args:
        splatter: Splatter instance (uses config to determine date)
        include: List of patterns to include (e.g., ['mesh'] for mesh files only)
        overwrite: Overwrite local files if they exist

    Returns:
        int: Number of files downloaded

    Example:
        >>> from collab_splats.wrapper import Splatter
        >>> from collab_splats.utils.gcs_helpers import pull_from_gcs
        >>> splatter = Splatter.from_config_file('config.yaml')
        >>> pull_from_gcs(splatter, include=['mesh'])
        >>> splatter.query_mesh(queries=['tree'])
    """
    from collab_data.gcs_utils import GCSClient
    from collab_data.file_utils import expand_path, get_project_root
    from dotenv import load_dotenv

    load_dotenv()

    # Init client
    key = Path(os.environ["COLLAB_DATA_KEY"])
    project_id = "-".join(key.stem.split("-")[:-1])
    client = GCSClient(
        project_id=project_id,
        credentials_path=expand_path(key.as_posix(), get_project_root())
    )

    # Get paths - extract date from file_path
    file_path = Path(splatter.config["file_path"])
    date = next(p for p in file_path.parts if len(p) == 10 and p.count("-") == 2)
    gcs_prefix = f"fieldwork_processed/processed_splats/{date}"
    output_path = Path(splatter.config["output_path"])

    print(f"Downloading from: {gcs_prefix}")
    if include:
        print(f"Including patterns: {include}")

    # Download
    gcs_files = client.glob(f"{gcs_prefix}/**")
    print(f"Found {len(gcs_files)} files in GCS")

    count = 0

    for gcs_path in gcs_files:
        rel = gcs_path.replace(f"{gcs_prefix}/", "")

        # Filter by include patterns
        if include and not any(pat in rel for pat in include):
            continue

        local = output_path / rel

        # Skip existing unless overwrite
        if local.exists() and not overwrite:
            print(f"⏭️  {rel}")
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"📥 {rel}")
        client.download_file(gcs_path, str(local))
        count += 1

    print(f"\n✅ Downloaded {count} files")
    return count
