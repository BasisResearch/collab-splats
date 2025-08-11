import os
from pathlib import Path
import shutil
from dotenv import load_dotenv

from collab_data.file_utils import expand_path, get_project_root
from collab_data.gcs_utils import GCSClient

SESSIONS = [
    "2023-11-05",
]

if __name__ == "__main__":

    load_dotenv()

    # Load environment variables from .env file
    data_key = os.environ.get("COLLAB_DATA_KEY")

    print(f"Data key: {data_key}")

    CURRENT_PROJECT = "COLLAB_DATA"
    PROJECT_KEY = Path(os.environ.get(f"{CURRENT_PROJECT}_KEY"))
    PROJECT_ID = "-".join(PROJECT_KEY.stem.split("-")[:-1])

    # Connect to GCS
    gcs_client = GCSClient(
        project_id=PROJECT_ID,
        credentials_path=expand_path(PROJECT_KEY.as_posix(), get_project_root()),
    )

    all_buckets = gcs_client.list_buckets()
    print(f"Available buckets: {all_buckets}")

    curated_data_dir = 'fieldwork_processed'
    
    # Find all files in the curated fieldwork
    curated_files = gcs_client.glob(f"{curated_data_dir}/*")

    for session in SESSIONS:
        
        session_dir = [d for d in curated_files if session.replace('-', '_') in d][0]

        print(f"Session dir: {session_dir}")    
        print(f"Session dir: {session_dir}")    



