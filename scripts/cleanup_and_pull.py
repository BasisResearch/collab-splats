#!/usr/bin/env python3
"""
Interactive script to clean up local files and pull from GCS.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from collab_data.file_utils import expand_path, get_project_root
from collab_data.gcs_utils import GCSClient


def init_gcs():
    """Initialize GCS client."""
    load_dotenv()
    key = Path(os.environ["COLLAB_DATA_KEY"])
    project_id = "-".join(key.stem.split("-")[:-1])
    return GCSClient(
        project_id=project_id,
        credentials_path=expand_path(key.as_posix(), get_project_root())
    )


def find_local_sessions():
    """Find all local sessions."""
    base = Path("/workspace/fieldwork-data")
    sessions = []

    for species_dir in base.iterdir():
        if species_dir.is_dir():
            for date_dir in species_dir.iterdir():
                if date_dir.is_dir() and len(date_dir.name) == 10:
                    env_dir = date_dir / "environment"
                    if env_dir.exists():
                        size = sum(f.stat().st_size for f in env_dir.rglob('*') if f.is_file())
                        sessions.append({
                            'date': date_dir.name,
                            'species': species_dir.name,
                            'path': date_dir,
                            'size_gb': size / (1024**3)
                        })

    return sorted(sessions, key=lambda x: x['date'])


def list_gcs_sessions(client):
    """List sessions on GCS."""
    sessions_gcs = client.glob("fieldwork_processed/processed_splats/*")
    sessions = []

    for session in sessions_gcs:
        date = session.split('/')[-1]
        files = client.glob(f"{session}/**")
        sessions.append({'date': date, 'file_count': len(files)})

    return sorted(sessions, key=lambda x: x['date'])


def main():
    print("Initializing GCS client...")
    client = init_gcs()

    print("\n" + "="*60)
    print("LOCAL SESSIONS")
    print("="*60)
    local = find_local_sessions()
    for s in local:
        print(f"  {s['date']} ({s['species']}): {s['size_gb']:.2f} GB")

    print("\n" + "="*60)
    print("GCS SESSIONS")
    print("="*60)
    gcs = list_gcs_sessions(client)
    for s in gcs:
        print(f"  {s['date']}: {s['file_count']} files")

    # Check which local sessions are backed up
    local_dates = {s['date'] for s in local}
    gcs_dates = {s['date'] for s in gcs}
    backed_up = local_dates & gcs_dates
    not_backed_up = local_dates - gcs_dates
    only_gcs = gcs_dates - local_dates

    print("\n" + "="*60)
    print("BACKED UP (can safely remove locally):")
    print("="*60)
    backed_up_sessions = [s for s in local if s['date'] in backed_up]
    total_size = sum(s['size_gb'] for s in backed_up_sessions)
    for s in backed_up_sessions:
        print(f"  {s['date']} ({s['species']}): {s['size_gb']:.2f} GB")
    print(f"\nTotal potential savings: {total_size:.2f} GB")

    if not_backed_up:
        print("\n⚠️  NOT BACKED UP (don't remove):")
        for date in not_backed_up:
            print(f"  {date}")

    if only_gcs:
        print("\n📥 AVAILABLE TO PULL FROM GCS:")
        for date in only_gcs:
            gcs_info = next(s for s in gcs if s['date'] == date)
            print(f"  {date}: {gcs_info['file_count']} files")

    # Interactive removal
    print("\n" + "="*60)
    if backed_up:
        response = input(f"\nRemove backed-up sessions locally? [y/N]: ")
        if response.lower() == 'y':
            for s in backed_up_sessions:
                env_dir = s['path'] / "environment"
                print(f"Removing {s['date']} ({s['size_gb']:.2f} GB)...")
                shutil.rmtree(env_dir)
                print(f"✅ Removed {s['date']}")

    # Interactive pull
    if only_gcs:
        print("\n" + "="*60)
        pull = input(f"\nWhich sessions to pull? (comma-separated dates or 'all'): ").strip()

        if pull.lower() == 'all':
            to_pull = list(only_gcs)
        elif pull:
            to_pull = [d.strip() for d in pull.split(',')]
        else:
            to_pull = []

        if to_pull:
            species = input(f"Species directory? (birds/rats/gerbils/ants): ").strip()
            include = input(f"Include patterns? (e.g., 'mesh preproc' or press enter for all): ").strip()
            include_patterns = include.split() if include else None

            for date in to_pull:
                if date not in only_gcs:
                    print(f"⚠️  {date} not available on GCS")
                    continue

                print(f"\nPulling {date}...")
                gcs_prefix = f"fieldwork_processed/processed_splats/{date}"
                local_base = Path(f"/workspace/fieldwork-data/{species}/{date}")
                local_base.mkdir(parents=True, exist_ok=True)

                gcs_files = client.glob(f"{gcs_prefix}/**")
                downloaded = 0

                for gcs_path in gcs_files:
                    rel = gcs_path.replace(f"{gcs_prefix}/", "")

                    if include_patterns and not any(pat in rel for pat in include_patterns):
                        continue

                    local_path = local_base / rel

                    if local_path.exists():
                        continue

                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(gcs_path, str(local_path))
                    downloaded += 1

                    if downloaded % 50 == 0:
                        print(f"  Downloaded {downloaded} files...")

                print(f"✅ Downloaded {downloaded} files for {date}")


if __name__ == "__main__":
    main()
