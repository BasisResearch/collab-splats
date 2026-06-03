"""Browse and transfer reconstruction sessions/videos to and from GCS via rclone."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from collab_data.data_dashboard.rclone_client import RcloneClient

logger = logging.getLogger(__name__)

########
# Constants
########

CURATED_BUCKET = "fieldwork_curated"
PROCESSED_BUCKET = "fieldwork_processed"
ROOT = "reconstruction"
_VIDEO_EXTS = (".mp4", ".mov")

########
# Source
########


class SessionSource:
    """Lists sessions/videos under fieldwork_curated/reconstruction and moves outputs."""

    def __init__(self, client: RcloneClient | None = None) -> None:
        self._client = client if client is not None else RcloneClient()

    def list_sessions(self) -> list[str]:
        """Return sorted YYYY_MM_DD session directory names."""
        items = self._client.list_directory(CURATED_BUCKET, ROOT)
        return sorted(i["Name"] for i in items if i.get("IsDir"))

    def list_videos(self, session: str) -> list[str]:
        """Return video filenames under a session."""
        items = self._client.list_directory(CURATED_BUCKET, f"{ROOT}/{session}")
        return [
            i["Name"]
            for i in items
            if not i.get("IsDir") and i["Name"].lower().endswith(_VIDEO_EXTS)
        ]

    def fetch_video(self, session: str, name: str, dest_dir: Path) -> Path:
        """rclone-copy a remote video to dest_dir; return the local path."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        remote = f"{self._client.remote_name}:{CURATED_BUCKET}/{ROOT}/{session}/{name}"
        subprocess.run(
            self._client._cmd("copyto", remote, str(local)), check=True
        )
        return local

    def has_processed(self, session: str, stem: str) -> bool:
        """True if processed outputs already exist for this video."""
        try:
            items = self._client.list_directory(
                PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}"
            )
        except Exception:  # rclone errors when the path does not exist
            return False
        return bool(items)

    def pull_processed(self, session: str, stem: str, dest_dir: Path) -> Path:
        """rclone-copy processed outputs to dest_dir; return the local dir."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{self._client.remote_name}:{PROCESSED_BUCKET}/{ROOT}/{session}/{stem}"
        subprocess.run(
            self._client._cmd("copy", remote, str(dest_dir)), check=True
        )
        return dest_dir

    def push_outputs(self, local_dir: Path, session: str, stem: str) -> None:
        """rclone-copy the full local output tree to fieldwork_processed."""
        self._client.copy_local_to_remote(
            str(local_dir), PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}"
        )
