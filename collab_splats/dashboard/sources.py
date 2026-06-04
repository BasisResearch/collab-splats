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
        if client is not None:
            self._client: RcloneClient | None = client
        else:
            try:
                self._client = RcloneClient()
            except Exception as exc:  # rclone missing/misconfigured -> degrade, surface on use
                logger.warning("rclone unavailable: %s", exc)
                self._client = None

    def _require_client(self) -> RcloneClient:
        """Return the rclone client, raising if unavailable."""
        if self._client is None:
            raise RuntimeError("rclone is not available")
        return self._client

    def list_sessions(self) -> list[str]:
        """Return sorted YYYY_MM_DD session directory names."""
        client = self._require_client()
        items = client.list_directory(CURATED_BUCKET, ROOT)
        return sorted(i["Name"] for i in items if i.get("IsDir"))

    def list_videos(self, session: str) -> list[str]:
        """Return video filenames under a session."""
        client = self._require_client()
        items = client.list_directory(CURATED_BUCKET, f"{ROOT}/{session}")
        return [i["Name"] for i in items if not i.get("IsDir") and i["Name"].lower().endswith(_VIDEO_EXTS)]

    def fetch_video(self, session: str, name: str, dest_dir: Path) -> Path:
        """rclone-copy a remote video to dest_dir; return the local path."""
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        remote = f"{client.remote_name}:{CURATED_BUCKET}/{ROOT}/{session}/{name}"
        subprocess.run(client._cmd("copyto", remote, str(local)), check=True)
        return local

    def has_processed(self, session: str, stem: str) -> bool:
        """True if processed outputs already exist for this video."""
        try:
            client = self._require_client()
            items = client.list_directory(PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}")
        except Exception:  # rclone errors when the path does not exist, or unavailable
            return False
        return bool(items)

    def pull_processed(self, session: str, stem: str, dest_dir: Path) -> Path:
        """rclone-copy processed outputs to dest_dir; return the local dir."""
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{client.remote_name}:{PROCESSED_BUCKET}/{ROOT}/{session}/{stem}"
        # no public remote->local API on RcloneClient; use _cmd directly
        subprocess.run(client._cmd("copy", remote, str(dest_dir)), check=True)
        return dest_dir

    def push_outputs(self, local_dir: Path, session: str, stem: str) -> None:
        """rclone-copy the full local output tree to fieldwork_processed."""
        client = self._require_client()
        client.copy_local_to_remote(str(local_dir), PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}")
