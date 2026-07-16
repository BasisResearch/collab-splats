"""Browse and transfer reconstruction sessions/videos to and from GCS via rclone."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable

from collab_data.data_dashboard.rclone_client import RcloneClient

logger = logging.getLogger(__name__)

########
# Constants
########

CURATED_BUCKET = "fieldwork_curated"
PROCESSED_BUCKET = "fieldwork_processed"
ROOT = "reconstruction"
_VIDEO_EXTS = (".mp4", ".mov")

# YYYY_MM_DD-session_XXXX field-session folders at the fieldwork_curated root
_FIELD_SESSION_RE = re.compile(r"^\d{4}_\d{2}_\d{2}-session_\d{4}$")

# Members skipped on every processed-scene pull: frames.zarr duplicates the frames/ jpg dir,
# and the dense per-pixel arrays are optional in FeedforwardResult.load_zarr (absent -> None)
# and unused by both the splats viewer and localization — they can be GBs per scene.
PULL_EXCLUDES = (
    "frames.zarr/**",
    "feedforward.zarr/depth/**",
    "feedforward.zarr/world_points/**",
    "feedforward.zarr/confidence/**",
    "feedforward.zarr/conf/**",  # legacy key for confidence
    "feedforward.zarr/features/**",
    "feedforward.zarr/pixel_indices/**",
    "feedforward.zarr/images/**",
)

########
# Helpers
########

# rclone --stats line looks like: "Transferred: 1.2 GiB / 5.6 GiB, 21%, 45 MiB/s, ETA 1m"
_RCLONE_PCT_RE = re.compile(r",\s*(\d{1,3})%")


def parse_rclone_percent(line: str) -> int | None:
    """Extract the integer transfer percentage from an rclone --stats line, or None."""
    m = _RCLONE_PCT_RE.search(line)
    if not m:
        return None
    return min(100, int(m.group(1)))


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

    def _run_streaming(self, cmd: list[str], action: str, on_line: Callable[[str], None] | None = None) -> None:
        """Run an rclone command via Popen, streaming --stats lines to on_line; raise on non-zero exit."""
        # `with` closes the stdout fd deterministically once streaming completes.
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout or []:
                line = line.strip()
                if line and on_line is not None:
                    on_line(line)
            ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"rclone {action} failed (exit {ret})")

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

    def fetch_video(
        self, session: str, name: str, dest_dir: Path, on_line: Callable[[str], None] | None = None
    ) -> Path:
        """rclone-copy a remote video to dest_dir; return the local path. on_line gets --stats lines."""
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        remote = f"{client.remote_name}:{CURATED_BUCKET}/{ROOT}/{session}/{name}"
        cmd = client._cmd("copyto", "--stats", "2s", "--stats-one-line", remote, str(local))
        self._run_streaming(cmd, f"copyto to {local}", on_line=on_line)
        return local

    def list_field_sessions(self) -> list[str]:
        """Return sorted YYYY_MM_DD-session_XXXX folders at the curated bucket root."""
        client = self._require_client()
        items = client.list_directory(CURATED_BUCKET, "")
        return sorted(i["Name"] for i in items if i.get("IsDir") and _FIELD_SESSION_RE.match(i["Name"]))

    def list_rgb_cameras(self, field_session: str) -> list[str]:
        """Return sorted rgb_X camera folders in a field session (thermal_X deferred)."""
        client = self._require_client()
        items = client.list_directory(CURATED_BUCKET, field_session)
        return sorted(i["Name"] for i in items if i.get("IsDir") and i["Name"].startswith("rgb_"))

    def list_camera_videos(self, field_session: str, camera: str) -> list[str]:
        """Return video filenames under a field session's camera folder."""
        client = self._require_client()
        items = client.list_directory(CURATED_BUCKET, f"{field_session}/{camera}")
        return [i["Name"] for i in items if not i.get("IsDir") and i["Name"].lower().endswith(_VIDEO_EXTS)]

    def fetch_field_video(self, field_session: str, camera: str, name: str, dest_dir: Path) -> Path:
        """rclone-copy a field-camera video to dest_dir; return the local path."""
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        remote = f"{client.remote_name}:{CURATED_BUCKET}/{field_session}/{camera}/{name}"
        subprocess.run(client._cmd("copyto", remote, str(local)), check=True)
        return local

    def list_localization_dbs(self, session: str, stem: str) -> list[str]:
        """Extractor names with a feature DB in the remote zarr (cheap directory listing)."""
        try:
            client = self._require_client()
            items = client.list_directory(PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}/feedforward.zarr/local_features")
        except Exception:  # path absent (no DB yet) or rclone unavailable
            return []
        return sorted(i["Name"] for i in items if i.get("IsDir"))

    def has_processed(self, session: str, stem: str) -> bool:
        """True if processed outputs already exist for this video."""
        try:
            client = self._require_client()
            items = client.list_directory(PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}")
        except Exception:  # rclone errors when the path does not exist, or unavailable
            return False
        return bool(items)

    def pull_processed(
        self,
        session: str,
        stem: str,
        dest_dir: Path,
        excludes: tuple = (),
        on_line: Callable[[str], None] | None = None,
    ) -> Path:
        """rclone-copy processed outputs to dest_dir; return the local dir.

        excludes: rclone --exclude patterns (e.g. "frames.zarr/**") to skip artifacts a
        consumer does not need. on_line, if given, receives each --stats progress line.
        """
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{client.remote_name}:{PROCESSED_BUCKET}/{ROOT}/{session}/{stem}"
        # Build --exclude flags, then stream one-line stats (no public remote->local API).
        flags: list[str] = []
        for pattern in excludes:
            flags += ["--exclude", pattern]
        flags += ["--stats", "2s", "--stats-one-line"]
        cmd = client._cmd("copy", *flags, remote, str(dest_dir))
        self._run_streaming(cmd, f"copy from {remote}", on_line=on_line)
        return dest_dir

    def push_outputs(
        self, local_dir: Path, session: str, stem: str, on_line: Callable[[str], None] | None = None
    ) -> None:
        """Stream the full local output tree to fieldwork_processed via `rclone copy`.

        Uses the recursive, idempotent `copy` verb (not file-only `copyto`) with
        rclone-native retries/timeouts and live one-line stats — streamed via Popen so a
        large tree (incl. frames.zarr) never trips an arbitrary wall-clock cap. on_line,
        if given, receives each progress line.
        """
        client = self._require_client()
        remote = f"{client.remote_name}:{PROCESSED_BUCKET}/{ROOT}/{session}/{stem}"
        cmd = client._cmd(
            "copy",
            "--gcs-bucket-policy-only",
            "--transfers",
            "8",
            "--retries",
            "3",
            "--timeout",
            "300s",
            "--contimeout",
            "60s",
            "--stats",
            "2s",
            "--stats-one-line",
            str(local_dir),
            remote,
        )
        self._run_streaming(cmd, f"copy to {remote}", on_line=on_line)
