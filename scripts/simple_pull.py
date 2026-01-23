#!/usr/bin/env python3
"""Simple script to pull specific sessions from GCS."""

import sys
from collab_splats.wrapper import Splatter
from collab_splats.utils import pull_from_gcs

# Available sessions on GCS that aren't local:
# 2024-05-19, 2024-05-23, 2024-05-27

SESSIONS = {
    "2024-05-19": "birds",  # Update species if different
    "2024-05-23": "birds",
    "2024-05-27": "birds",
}

if len(sys.argv) < 2:
    print("Usage: python simple_pull.py <date> [include_pattern]")
    print("\nAvailable dates:")
    for date, species in SESSIONS.items():
        print(f"  {date} ({species})")
    print("\nExamples:")
    print("  python simple_pull.py 2024-05-19")
    print("  python simple_pull.py 2024-05-19 mesh  # Only download mesh files")
    sys.exit(1)

date = sys.argv[1]
include = sys.argv[2:] if len(sys.argv) > 2 else None

if date not in SESSIONS:
    print(f"❌ Date {date} not available. Choose from: {list(SESSIONS.keys())}")
    sys.exit(1)

species = SESSIONS[date]

# Create a dummy config to use pull_from_gcs
print(f"Pulling {date} ({species}) from GCS...")
if include:
    print(f"Including patterns: {include}")

# We need to create a Splatter config
from pathlib import Path

config_dict = {
    "file_path": Path(f"/workspace/fieldwork-data/{species}/{date}/dummy.mp4"),
    "method": "rade-features",
    "output_path": Path(f"/workspace/fieldwork-data/{species}/{date}"),
}

# Create splatter instance with minimal config
class FakeSplatter:
    def __init__(self, config):
        self.config = config

splatter = FakeSplatter(config_dict)
downloaded = pull_from_gcs(splatter, include=include)

print(f"\n✅ Done! Downloaded {downloaded} files")
