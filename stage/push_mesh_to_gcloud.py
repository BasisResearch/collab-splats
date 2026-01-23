#!/usr/bin/env python3
"""
Script to push mesh solution directories to Google Cloud Storage.

Usage:
    python push_mesh_to_gcloud.py --session rats_date-07112024_video-C0119
    python push_mesh_to_gcloud.py --session birds_date-02062024_video-C0043
    python push_mesh_to_gcloud.py --all  # Push all pending sessions
"""

import os
from pathlib import Path
from typing import List, Optional
import argparse
from dotenv import load_dotenv

from collab_data.file_utils import expand_path, get_project_root
from collab_data.gcs_utils import GCSClient


def parse_session_name(session_name: str) -> dict:
    """
    Parse session name into components.

    Example: 'rats_date-07112024_video-C0119' ->
        {
            'species': 'rats',
            'date': '2024-07-11',
            'video': 'C0119'
        }
    """
    parts = session_name.split('_')
    species = parts[0]

    # Extract date (format: MMDDYYYY)
    date_str = parts[1].replace('date-', '')
    month = date_str[:2]
    day = date_str[2:4]
    year = date_str[4:]
    date = f"{year}-{month}-{day}"

    # Extract video ID
    video = parts[2].replace('video-', '')

    return {
        'species': species,
        'date': date,
        'video': video
    }


def get_local_path(session_info: dict) -> Path:
    """Get local path for a session."""
    species = session_info['species']
    date = session_info['date']
    video = session_info['video']

    local_path = Path(f"/workspace/fieldwork-data/{species}/{date}/environment/{video}")
    return local_path


def get_gcs_path(session_info: dict) -> str:
    """Get GCS path for a session."""
    date = session_info['date']  # Keep hyphens in date
    return f"fieldwork_processed/processed_splats/{date}"


def push_directory_to_gcs(
    gcs_client: GCSClient,
    local_dir: Path,
    gcs_prefix: str,
    skip_existing: bool = True,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> int:
    """
    Push a local directory to GCS.

    Args:
        gcs_client: Initialized GCS client
        local_dir: Local directory path
        gcs_prefix: GCS prefix (bucket will be prepended automatically)
        skip_existing: If True, skip files that already exist in GCS
        include_patterns: List of patterns to include (e.g., ['mesh', 'rade-features'])
        exclude_patterns: List of patterns to exclude (e.g., ['images_4', 'images_8'])

    Returns:
        Number of files uploaded
    """
    if not local_dir.exists():
        raise ValueError(f"Local directory does not exist: {local_dir}")

    # Get existing files in GCS if skip_existing is True
    existing_files = set()
    if skip_existing:
        existing_files = set(gcs_client.glob(f"{gcs_prefix}/**"))
        print(f"Found {len(existing_files)} existing files in GCS")

    uploaded_count = 0

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(local_dir)

            # Check exclude patterns
            if exclude_patterns:
                if any(pattern in str(relative_path) for pattern in exclude_patterns):
                    continue

            # Check include patterns (if specified, at least one must match)
            if include_patterns:
                if not any(pattern in str(relative_path) for pattern in include_patterns):
                    continue

            gcs_path = f"{gcs_prefix}/{relative_path.as_posix()}"

            # Skip if file already exists and skip_existing is True
            if skip_existing and gcs_path in existing_files:
                print(f"⏭️  Skipping (exists): {relative_path}")
                continue

            # Upload the file
            gcs_client.upload_file(str(local_path), gcs_path)
            uploaded_count += 1

    return uploaded_count


def main():
    parser = argparse.ArgumentParser(description='Push mesh solutions to Google Cloud Storage')
    parser.add_argument(
        '--session',
        type=str,
        help='Session name (e.g., rats_date-07112024_video-C0119)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Push all sessions listed in the script'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip files that already exist in GCS (default: True)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force upload all files, even if they exist'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize GCS client
    CURRENT_PROJECT = "COLLAB_DATA"
    PROJECT_KEY = Path(os.environ.get(f"{CURRENT_PROJECT}_KEY"))
    PROJECT_ID = "-".join(PROJECT_KEY.stem.split("-")[:-1])

    gcs_client = GCSClient(
        project_id=PROJECT_ID,
        credentials_path=expand_path(PROJECT_KEY.as_posix(), get_project_root()),
    )

    # Define sessions to push
    sessions = []
    if args.all:
        sessions = [
            "rats_date-07112024_video-C0119",
            "birds_date-02062024_video-C0043"
        ]
    elif args.session:
        sessions = [args.session]
    else:
        parser.error("Must specify either --session or --all")

    # Process each session
    for session_name in sessions:
        print(f"\n{'='*60}")
        print(f"Processing: {session_name}")
        print(f"{'='*60}")

        session_info = parse_session_name(session_name)
        local_dir = get_local_path(session_info)
        gcs_prefix = get_gcs_path(session_info)

        print(f"Species: {session_info['species']}")
        print(f"Date: {session_info['date']}")
        print(f"Video: {session_info['video']}")
        print(f"Local path: {local_dir}")
        print(f"GCS path: {gcs_prefix}")
        print()

        if not local_dir.exists():
            print(f"❌ Error: Local directory not found: {local_dir}")
            continue

        # Push the entire directory
        uploaded = push_directory_to_gcs(
            gcs_client=gcs_client,
            local_dir=local_dir,
            gcs_prefix=gcs_prefix,
            skip_existing=not args.force,
            exclude_patterns=['images_4', 'images_8']  # Exclude downsampled images
        )

        print(f"\n✅ Uploaded {uploaded} files for {session_name}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
