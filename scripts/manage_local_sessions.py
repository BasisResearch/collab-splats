#!/usr/bin/env python3
"""
Manage local session data - remove pushed sessions and pull others from GCS.

Usage:
    # Remove local files (after confirming they're backed up on GCS)
    python manage_local_sessions.py --remove 2024-02-06 2024-07-11

    # Pull sessions from GCS
    python manage_local_sessions.py --pull 2024-05-19 2024-05-23

    # Pull specific patterns only
    python manage_local_sessions.py --pull 2024-05-27 --include mesh preproc
"""

import os
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from collab_data.file_utils import expand_path, get_project_root
from collab_data.gcs_utils import GCSClient


def find_local_path(date: str) -> Path:
    """Find local path for a date (searches common species directories)."""
    base = Path("/workspace/fieldwork-data")

    # Common species directories
    species = ["birds", "rats", "gerbils", "ants"]

    for sp in species:
        path = base / sp / date
        if path.exists():
            return path

    return None


def verify_gcs_backup(gcs_client, date: str) -> bool:
    """Verify that files exist on GCS before removing locally."""
    gcs_prefix = f"fieldwork_processed/processed_splats/{date}"
    files = gcs_client.glob(f"{gcs_prefix}/**")
    return len(files) > 0


def remove_local_session(gcs_client, date: str, force: bool = False):
    """Remove local session files after verifying GCS backup."""
    local_path = find_local_path(date)

    if local_path is None:
        print(f"❌ Local path not found for {date}")
        return False

    env_dir = local_path / "environment"
    if not env_dir.exists():
        print(f"❌ Environment directory not found: {env_dir}")
        return False

    # Verify GCS backup
    if not force:
        print(f"Verifying GCS backup for {date}...")
        if not verify_gcs_backup(gcs_client, date):
            print(f"❌ No backup found on GCS for {date}. Aborting.")
            return False
        print(f"✅ GCS backup verified")

    # List what will be removed
    total_size = sum(f.stat().st_size for f in env_dir.rglob('*') if f.is_file())
    total_size_gb = total_size / (1024**3)

    print(f"\nWill remove: {env_dir}")
    print(f"Size: {total_size_gb:.2f} GB")

    if not force:
        response = input(f"Remove {date} from local storage? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return False

    # Remove the directory
    print(f"Removing {env_dir}...")
    shutil.rmtree(env_dir)
    print(f"✅ Removed {date}")

    return True


def pull_session(gcs_client, date: str, include_patterns: list = None):
    """Pull session from GCS."""
    gcs_prefix = f"fieldwork_processed/processed_splats/{date}"

    # Determine local path (need to figure out species)
    # For now, check what's on GCS to determine structure
    print(f"Pulling {date} from GCS...")
    gcs_files = gcs_client.glob(f"{gcs_prefix}/**")

    if len(gcs_files) == 0:
        print(f"❌ No files found on GCS for {date}")
        return False

    print(f"Found {len(gcs_files)} files on GCS")

    # Try to determine species from first file
    # The structure is: fieldwork_processed/processed_splats/DATE/environment/VIDEO/...
    # But we need to know species to download to correct location

    # Let's look at existing local structure to determine species
    base = Path("/workspace/fieldwork-data")
    species_dirs = [d for d in base.iterdir() if d.is_dir()]

    print(f"\nAvailable species directories: {[d.name for d in species_dirs]}")
    species = input(f"Which species directory for {date}? (birds/rats/gerbils/ants): ").strip()

    if not species:
        print("Cancelled.")
        return False

    local_base = base / species / date
    local_base.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    for gcs_path in gcs_files:
        # Extract relative path
        rel = gcs_path.replace(f"{gcs_prefix}/", "")

        # Filter by include patterns
        if include_patterns and not any(pat in rel for pat in include_patterns):
            continue

        local_path = local_base / rel

        # Skip if exists
        if local_path.exists():
            print(f"⏭️  {rel}")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"📥 {rel}")
        gcs_client.download_file(gcs_path, str(local_path))
        downloaded += 1

    print(f"\n✅ Downloaded {downloaded} files to {local_base}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Manage local session data')
    parser.add_argument(
        '--remove',
        nargs='+',
        metavar='DATE',
        help='Remove local sessions (format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--pull',
        nargs='+',
        metavar='DATE',
        help='Pull sessions from GCS (format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--include',
        nargs='+',
        help='Patterns to include when pulling (e.g., mesh preproc)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )

    args = parser.parse_args()

    if not args.remove and not args.pull:
        parser.error("Must specify --remove or --pull")

    # Load environment and init GCS client
    load_dotenv()

    project_key = Path(os.environ.get("COLLAB_DATA_KEY"))
    project_id = "-".join(project_key.stem.split("-")[:-1])
    gcs_client = GCSClient(
        project_id=project_id,
        credentials_path=expand_path(project_key.as_posix(), get_project_root()),
    )

    # Remove local sessions
    if args.remove:
        print("="*60)
        print("REMOVING LOCAL SESSIONS")
        print("="*60)
        for date in args.remove:
            print(f"\n--- {date} ---")
            remove_local_session(gcs_client, date, args.force)

    # Pull sessions
    if args.pull:
        print("\n" + "="*60)
        print("PULLING SESSIONS FROM GCS")
        print("="*60)
        for date in args.pull:
            print(f"\n--- {date} ---")
            pull_session(gcs_client, date, args.include)


if __name__ == "__main__":
    main()
