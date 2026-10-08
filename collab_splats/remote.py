"""
Scene-level access to the curated and processed GCS buckets.

- a scene id is the curated dir name; its processed outputs live under the same name
- transport is collab-data's RcloneClient; this module knows buckets, ids and excludes
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from collab_data.data_dashboard.rclone_client import STATS_ARGS, RcloneClient

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# One path segment, no leading "." or "-": ids are joined onto local output paths
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Never pushed: caches, build databases, source video; images/ is pushed, so images/ patterns need a leading slash
PUSH_EXCLUDES = (
    # Scene-root temporary full-width features; the 2D codes and <backend>/semantics are pushed
    "/semantics/*_features.zarr/**",
    # Source video, already in the curated bucket; bracketed because rclone globs are case sensitive
    "*.[Mm][Pp]4",
    "*.[Mm][Oo][Vv]",
    "*.[Aa][Vv][Ii]",
    # SIFT databases and hloc intermediates, rebuilt from images/
    "/*/colmap/instantsfm.db",
    "/*/colmap/colmap.db",
    "/*/colmap/hloc/**",
)

########################################
# Source
########################################


class SceneSource:
    """
    Curated scene listing and processed-output transfers for one rclone remote.

    - listings are memoized for listing_ttl seconds; transfers invalidate what they change
    - an absent GCS prefix lists as []; any rclone failure raises, so absent never looks like unreachable
    """

    def __init__(
        self,
        client: RcloneClient | None = None,
        *,
        curated: str = "environments-curated",
        processed: str = "environments-processed",
        video_exts: tuple[str, ...] = (".mp4", ".mov", ".avi"),
        listing_ttl: float = 60.0,
    ) -> None:
        """
        Bind a client and the two buckets.

        - a client that fails to construct degrades to None; check_available retries it

        Args:
            client: rclone client; None builds the default one.
            curated: bucket of input scenes, one video per scene dir.
            processed: bucket receiving pipeline outputs.
            video_exts: lower-case extensions accepted as a scene's video.
            listing_ttl: seconds a memoized listing stays fresh.
        """
        self.curated = curated
        self.processed = processed
        self.video_exts = video_exts
        self.listing_ttl = listing_ttl
        self._listing_cache: dict = {}

        # Build the default client; a broken rclone surfaces on first use, not here
        self._client = client

        if client is None:
            try:
                self._client = RcloneClient()
            except Exception as exc:
                logger.warning("rclone unavailable: %s", exc)

    def _require_client(self) -> RcloneClient:
        """
        The client, or RuntimeError when rclone never came up.
        """
        if self._client is None:
            raise RuntimeError("rclone is not available")

        return self._client

    def _path(self, bucket: str, path: str = "") -> str:
        """
        `<remote>:<bucket>[/<path>]` for this client's remote.
        """
        remote = f"{self._require_client().remote_name}:{bucket}"
        return f"{remote}/{path.strip('/')}" if path else remote

    def _lsjson(self, bucket: str, path: str = "") -> list[dict]:
        """
        Entries under a bucket path; [] when absent, RuntimeError when rclone fails.

        - GCS prefixes are not directories, so an absent scene lists with exit 0 and []
        - a non-zero exit, exit 3 included, means a missing bucket or a fault, never absence
        """
        out = self._require_client().run("lsjson", self._path(bucket, path)).strip()
        return json.loads(out) if out else []

    def _cached(self, key: tuple, producer: Callable[[], Any]) -> Any:
        """
        Memoized listing for key, refreshed once the TTL has elapsed.

        - cached values are shared by reference; callers must not mutate them
        """
        now = time.monotonic()
        hit = self._listing_cache.get(key)

        if hit is not None and hit[0] > now:
            return hit[1]

        # Drop other expired entries so the memo does not grow unbounded
        expired = [k for k, (exp, _v) in self._listing_cache.items() if exp <= now]

        for k in expired:
            del self._listing_cache[k]

        # Produce and memoize a fresh value
        value = producer()
        self._listing_cache[key] = (now + self.listing_ttl, value)
        return value

    def invalidate(self, key: tuple | None = None) -> None:
        """
        Drop one memoized listing, or all of them.

        Args:
            key: the cache key to drop; None clears the whole cache.
        """
        if key is None:
            self._listing_cache.clear()
        else:
            self._listing_cache.pop(key, None)

    def check_available(self) -> bool:
        """
        Uncached liveness probe against the curated bucket.

        - tells "this scene failed" apart from "rclone stopped working"
        - bypasses the listing memo, so a stale success cannot answer
        - retries client construction when rclone was unavailable at init
        - an empty bucket still counts as available
        - never raises

        Returns:
            True when rclone can list the curated bucket right now.
        """
        # Retry a client that failed at construction, or a transient fault would stick forever
        if self._client is None:
            try:
                self._client = RcloneClient()
            except Exception as exc:
                logger.error("rclone still unavailable on re-attempt: %s", exc)
                return False

        # Catch-all on purpose: the probe must answer, never raise the fault it detects
        try:
            self._lsjson(self.curated, "")
        except Exception as exc:
            logger.error("rclone liveness probe failed against %s: %s", self.curated, exc)
            return False

        return True

    ########################################
    # Curated inputs
    ########################################

    def list_scenes(self) -> list[str]:
        """
        Scene ids in the curated bucket.

        - dirs failing SCENE_ID_RE are skipped with a warning
        - an unreachable bucket raises instead of listing empty

        Returns:
            Sorted dir names directly under the curated bucket.
        """

        def _produce() -> list[str]:
            entries = self._lsjson(self.curated, "")

            if not entries:
                logger.info("%s listed empty — no curated scenes to process", self.curated)
                return []

            # Name skipped dirs, or a naming drift silently drops scenes from --all
            dirs = [e["Name"] for e in entries if e.get("IsDir")]
            scenes = []
            skipped = []

            for d in sorted(dirs):
                (scenes if SCENE_ID_RE.match(d) else skipped).append(d)

            if skipped:
                listed = ", ".join(skipped)
                logger.warning("skipping non-scene dirs in %s: %s", self.curated, listed)

            return scenes

        return self._cached(("list_scenes",), _produce)

    def scene_video(self, scene: str) -> str:
        """
        Video filename inside a curated scene dir.

        - several videos: warns and takes the first by name

        Args:
            scene: curated scene id.

        Returns:
            The video's filename, not a path.

        Raises:
            FileNotFoundError: the scene dir holds no video.
        """

        def _produce() -> list[str]:
            # Video files in the scene dir, by name
            entries = self._lsjson(self.curated, scene)
            videos = sorted(
                e["Name"] for e in entries if not e.get("IsDir") and e["Name"].lower().endswith(self.video_exts)
            )

            # Reported inside the producer so it fires once per TTL, not on every cached hit
            if not videos:
                logger.info("no video in %s/%s — skipping", self.curated, scene)

            return videos

        # Raising outside the producer memoizes the listing, while a fault never reaches the memo
        videos = self._cached(("scene_video", scene), _produce)

        if not videos:
            raise FileNotFoundError(f"no video in {self.curated}/{scene}")

        if len(videos) > 1:
            logger.warning("%s holds %d videos; using %s", scene, len(videos), videos[0])

        return videos[0]

    def fetch_video(self, scene: str, dest_dir: Path, on_line: Callable[[str], None] | None = None) -> Path:
        """
        Copy the scene's video into dest_dir.

        Args:
            scene: curated scene id.
            dest_dir: local directory, created if absent.
            on_line: receives each rclone output line.

        Returns:
            Local path of the video.
        """
        # Resolve the video name and create the destination
        name = self.scene_video(scene)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy the one remote file to its local path
        local = dest_dir / name
        remote = self._path(self.curated, f"{scene}/{name}")
        self._require_client().run_streaming("copyto", remote, str(local), *STATS_ARGS, on_line=on_line)
        return local

    ########################################
    # Processed state
    ########################################

    def list_processed_scenes(self) -> list[str]:
        """
        Scene ids with processed outputs.

        Returns:
            Sorted dir names under the processed bucket that match SCENE_ID_RE.
        """

        # Every processed dir comes from push_outputs, so a non-matching name is a stray upload
        def _produce() -> list[str]:
            entries = self._lsjson(self.processed, "")

            if not entries:
                logger.info("%s listed empty — nothing processed yet", self.processed)
                return []

            return sorted(e["Name"] for e in entries if e.get("IsDir") and SCENE_ID_RE.match(e["Name"]))

        return self._cached(("list_processed_scenes",), _produce)

    def has_processed(self, scene: str) -> bool:
        """
        Whether a scene already has processed outputs.

        Args:
            scene: processed scene id.

        Returns:
            True when the scene's processed dir lists non-empty.
        """

        def _produce() -> bool:
            entries = self._lsjson(self.processed, scene)

            if not entries:
                logger.info("%s has no processed outputs yet — %s listed empty", scene, self.processed)

            return bool(entries)

        return self._cached(("has_processed", scene), _produce)

    ########################################
    # Transfers
    ########################################

    def pull_processed(
        self,
        scene: str,
        dest_dir: Path,
        excludes: tuple[str, ...] = (),
        on_line: Callable[[str], None] | None = None,
    ) -> Path:
        """
        Copy a scene's processed outputs into dest_dir.

        Args:
            scene: processed scene id.
            dest_dir: local directory, created if absent.
            excludes: rclone --exclude globs to skip.
            on_line: receives each rclone output line.

        Returns:
            dest_dir.
        """
        # Create the destination, then copy the scene dir minus excludes
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = self._path(self.processed, scene)
        self._require_client().copy_dir(remote, str(dest_dir), exclude=excludes, on_line=on_line)

        return dest_dir

    def push_outputs(
        self,
        local_dir: Path,
        scene: str,
        on_line: Callable[[str], None] | None = None,
        *,
        transfers: int = 8,
        retries: int = 3,
        timeout: str = "300s",
        contimeout: str = "60s",
    ) -> None:
        """
        Upload a scene directory to the processed bucket, minus PUSH_EXCLUDES.

        Args:
            local_dir: local scene directory.
            scene: processed scene id.
            on_line: receives each rclone output line.
            transfers: parallel file transfers.
            retries: rclone retries of a failed copy.
            timeout: rclone IO idle timeout, as an rclone duration.
            contimeout: rclone connect timeout, as an rclone duration.
        """
        # rclone tuning flags, then copy the scene dir minus excludes
        flags = ["--gcs-bucket-policy-only", "--transfers", str(transfers), "--retries", str(retries)]
        flags += ["--timeout", timeout, "--contimeout", contimeout]
        remote = self._path(self.processed, scene)
        self._require_client().copy_dir(str(local_dir), remote, exclude=PUSH_EXCLUDES, on_line=on_line, extra=flags)

        # Drop memoized listings the push made stale
        self.invalidate(("has_processed", scene))
        self.invalidate(("list_processed_scenes",))

    def verify_push(self, local_dir: Path, scene: str, *, checkers: int = 16) -> bool:
        """
        True when every pushed local file exists remotely with equal content.

        - gates the remote driver's local delete, so any failure answers False

        Args:
            local_dir: local scene directory.
            scene: processed scene id.
            checkers: parallel rclone checkers.

        Returns:
            False on a mismatch or when the check could not run.
        """
        # rclone check flags; log what is being verified
        flags = ["--gcs-bucket-policy-only", "--fast-list", "--checkers", str(checkers)]
        logger.info("verifying %s against %s/%s", local_dir, self.processed, scene)

        # Any fault answers False: cannot-verify must never read as verified
        try:
            remote = self._path(self.processed, scene)
            matched = self._require_client().check(str(local_dir), remote, exclude=PUSH_EXCLUDES, extra=flags)
        except (RuntimeError, OSError) as exc:
            logger.error("verify could not run for %s, local data kept: %s", scene, exc)
            return False

        if not matched:
            logger.error("verify FAILED for %s: content mismatch, local data kept", scene)

        return matched
