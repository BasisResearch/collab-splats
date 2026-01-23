#!/usr/bin/env python3
"""List all sessions available in GCS."""

import os
from pathlib import Path
from dotenv import load_dotenv
from collab_data.file_utils import expand_path, get_project_root
from collab_data.gcs_utils import GCSClient

load_dotenv()

# Initialize GCS client
PROJECT_KEY = Path(os.environ.get("COLLAB_DATA_KEY"))
PROJECT_ID = "-".join(PROJECT_KEY.stem.split("-")[:-1])

gcs_client = GCSClient(
    project_id=PROJECT_ID,
    credentials_path=expand_path(PROJECT_KEY.as_posix(), get_project_root()),
)

# List all sessions in processed_splats
print("Available sessions in GCS (fieldwork_processed/processed_splats/):\n")
sessions = gcs_client.glob("fieldwork_processed/processed_splats/*")

for session in sorted(sessions):
    date = session.split('/')[-1]
    # Count files
    files = gcs_client.glob(f"{session}/**")
    print(f"  {date}: {len(files)} files")

print("\nTotal sessions:", len(sessions))
