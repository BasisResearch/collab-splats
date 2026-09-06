"""rclone-backed access to the environments-curated / environments-processed GCS buckets."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Callable

from collab_data.data_dashboard.rclone_client import RcloneClient

logger = logging.getLogger(__name__)

########
# Constants
########

CURATED_BUCKET = "environments-curated"
PROCESSED_BUCKET = "environments-processed"

# Extensions accepted as *the* video inside a curated scene dir. Deliberately separate from
# wrapper.batch.VIDEO_EXTS: that one scans a local filesystem, this one filters a bucket listing.
_VIDEO_EXTS = (".mp4", ".mov", ".avi")

# The scene id IS the curated dir name and is joined onto a local output path by the remote
# driver, so this regex enforces PATH SAFETY, not a naming shape: one path segment (no "/"),
# no leading "." (kills "..", ".", hidden dirs), no leading "-" (argv-safe). The historical
# YYYY_MM_DD-PARENTFOLDER-VIDEONAME shape is a convention some dirs follow, not the contract —
# audiomoth deployments ship names like audiomoth_only_deployments-<range>-<site>-...-<video>.
# Public because discovery and the driver's explicit-id check must accept exactly the same ids.
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Dense per-frame arrays a viewer does not need — pulled on demand instead.
PULL_EXCLUDES = (
    "pointcloud.zarr/depth/**",
    "pointcloud.zarr/world_points/**",
    "pointcloud.zarr/confidence/**",
    "pointcloud.zarr/conf/**",  # legacy key for confidence
    "pointcloud.zarr/features/**",
    "pointcloud.zarr/pixel_indices/**",
    "pointcloud.zarr/images/**",
)

# Never pushed: raw 2D feature maps (regenerable from frames + extractor) and the source
# video itself — the remote driver fetches it into the same scene dir it later pushes, and
# it already lives in environments-curated, so uploading it would just duplicate the
# largest file in the tree and make every verify re-hash it.
# The scene-root images/ store IS pushed: it is the sole persistent keyframe store, so a pulled
# processed scene must carry it to be localizable without re-running preproc — which is exactly
# what PULL_EXCLUDES above already assumes. Excluding it here made pulling it impossible.
# PULL_EXCLUDES' "pointcloud.zarr/images/**" does not reach it: that pattern has two path
# components, so it needs a pointcloud.zarr parent. A BARE "images/**" in either tuple would —
# see the anchoring note below — so anything added for images/ must carry a leading slash.
# Bracket classes rather than bare `*.mp4`: rclone globs are case sensitive and camera
# files are commonly uppercase (C0043.MP4).
# The 2D-cache pattern MUST keep its leading slash. rclone matches an unanchored pattern at ANY
# depth (measured, v1.53.3-DEV: `--exclude 'features/**'` on a local copy also skipped
# `sub/pointcloud.zarr/features/x`). Unanchored, `semantics/**` would also match
# `<backend>/semantics/**` — the lifted per-point features, which are the whole point of the push —
# and it would still match the pointcloud.zarr `features` member. Anchored it hits exactly
# Reconstructor.semantics_cache_dir, the regenerable patch cache at the pushed root.
PUSH_EXCLUDES = (
    "/semantics/**",
    "*.[Mm][Pp]4",
    "*.[Mm][Oo][Vv]",
    "*.[Aa][Vv][Ii]",
    # COLMAP match database is a local build artifact, rebuildable from the zarr feature
    # cache + poses (geometry/verification.py). Anchored at <backend>/colmap depth.
    "/*/colmap/database.db",
    # InstantSfM's SIFT database (pointcloud/sfm.py) — same class of artifact, its own name so
    # it never collides with the verification DB above. Rebuilt from the staged images.
    "/*/colmap/instantsfm.db",
)

########
# Helpers
########

# rclone --stats line looks like: "Transferred: 1.2 GiB / 5.6 GiB, 21%, 45 MiB/s, ETA 1m"
_RCLONE_PCT_RE = re.compile(r",\s*(\d{1,3})%")

# rclone check returns exit 1 for *every* failure — a content difference, a missing destination and
# an unconfigured remote alike (verified, rclone v1.53.3-DEV) — so the exit code cannot tell a real
# mismatch from a check that never ran. rclone's own summary line can: a genuine difference always
# reports "N differences found", while a transport/config fault dies before comparing anything
# ("Failed to create file system for ..."). The marker reaches _run_streaming's bounded tail because
# rclone's fatal exit line restates it ("Failed to check with N errors: last error was: ..."), not
# because the summary is always the last line — up to four NOTICE lines can follow it.
_CHECK_MISMATCH_MARKER = "differences found"


def _stats_args(interval: str) -> list[str]:
    """rclone flags that make --stats progress lines actually reach stdout at the given interval.

    --stats on its own emits nothing: rclone logs stats at INFO while its default --log-level is
    NOTICE, so every interval line is filtered out before it leaves the process (verified against
    rclone v1.53.3-DEV — a 1.5 GB copy printed no progress at all until --stats-log-level was
    raised). Any command whose progress feeds on_line/parse_rclone_percent needs this, not bare
    --stats, or its callback is silent for the entire transfer.
    """
    return ["--stats", interval, "--stats-one-line", "--stats-log-level", "NOTICE"]


def parse_rclone_percent(line: str) -> int | None:
    """Extract the integer transfer percentage from an rclone --stats line, or None."""
    m = _RCLONE_PCT_RE.search(line)
    if not m:
        return None
    return min(100, int(m.group(1)))


########
# Source
########


class SceneSource:
    """Lists flat scene dirs in environments-curated and moves outputs to environments-processed.

    A scene id is a single string — the curated dir name (YYYY_MM_DD-PARENTFOLDER-VIDEONAME),
    which holds exactly one video. Processed outputs live at environments-processed/<scene>/.
    """

    def __init__(self, client: RcloneClient | None = None) -> None:
        if client is not None:
            self._client: RcloneClient | None = client
        else:
            try:
                self._client = RcloneClient()
            except Exception as exc:  # rclone missing/misconfigured -> degrade, surface on use
                logger.warning("rclone unavailable: %s", exc)
                self._client = None
        # (key -> (expiry_epoch, value)) memo for cheap-but-repeated rclone directory listings.
        self._listing_cache: dict = {}
        self._listing_ttl = 60.0  # seconds; bucket contents rarely change mid-session

    def _require_client(self) -> RcloneClient:
        """Return the rclone client, raising if unavailable."""
        if self._client is None:
            raise RuntimeError("rclone is not available")
        return self._client

    @property
    def _remote(self) -> str:
        """rclone remote name — the single source for the `<remote>:` prefix on every path."""
        return self._require_client().remote_name

    def _cached(self, key: tuple, producer):
        """Return a memoized listing for key, refreshing when the TTL has elapsed.

        Cached values are shared by reference — callers must not mutate returned lists.
        """
        now = time.monotonic()
        hit = self._listing_cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        # Refresh path: drop other expired entries so the memo doesn't grow unbounded.
        for k in [k for k, (exp, _v) in self._listing_cache.items() if exp <= now]:
            del self._listing_cache[k]
        value = producer()
        self._listing_cache[key] = (now + self._listing_ttl, value)
        return value

    def invalidate(self, key: tuple | None = None) -> None:
        """Drop one cached listing (by key) or the whole listing cache."""
        if key is None:
            self._listing_cache.clear()
        else:
            self._listing_cache.pop(key, None)

    def _run_streaming(self, cmd: list[str], action: str, on_line: Callable[[str], None] | None = None) -> None:
        """Run an rclone command via Popen, streaming --stats lines to on_line; raise on non-zero exit."""
        # stderr is merged into stdout because that is where rclone writes --stats; keep the last
        # few lines so the failure carries rclone's own words (auth/403 etc.) to an aborting caller.
        tail: deque[str] = deque(maxlen=5)
        # `with` closes the stdout fd deterministically once streaming completes.
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout or []:
                line = line.strip()
                if not line:
                    continue
                # Progress never explains a failure but arrives far more often than errors do:
                # --stats 2s over a --retries 3 transfer emits enough interval lines to push an
                # early per-object ERROR out of a 5-line tail. Excluded from the tail only —
                # on_line still sees every line, because progress is what it exists for.
                if parse_rclone_percent(line) is None:
                    tail.append(line)
                if on_line is not None:
                    on_line(line)
            ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"rclone {action} failed (exit {ret}): {' | '.join(tail)}")

    def _lsjson(self, bucket: str, path: str) -> list[dict]:
        """List a remote dir, returning its entries; raise RuntimeError on ANY non-zero rclone exit.

        Absence is *not* signalled by an exit code here. Both buckets live on GCS, where a prefix is
        not a directory, so an absent scene lists successfully with an empty array — verified against
        rclone v1.53.3-DEV: `lsjson <bucket>/no-such-scene` exits 0 with `[]`, while a missing or
        misspelled *bucket* exits 3. CURATED_BUCKET and PROCESSED_BUCKET are fixed constants, so
        that exit 3 can only ever be a config fault, never "this scene is unprocessed" — treating it
        as absence is what would let an unattended driver re-reconstruct an entire bucket. Callers
        therefore read real absence off an empty return, and any raise means "answer unknown, abort".

        On a backend where absence itself returns 3 or 4 (a plain local filesystem, say) this
        classification would need revisiting.

        RcloneClient.list_directory collapses every failure — missing dir, 403, missing binary —
        into an empty list, so it cannot back a listing anyone reasons about: absent and unreachable
        must not look alike. This runs lsjson through the client's own argv builder (the same _cmd +
        subprocess pattern the transfers already use) so the exit code survives.
        """
        remote = f"{self._remote}:{bucket}"
        if path:
            remote += f"/{path.strip('/')}"
        try:
            result = subprocess.run(
                self._require_client()._cmd("lsjson", remote),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"rclone lsjson {remote} failed (exit {exc.returncode}): {detail}") from exc
        out = result.stdout.strip()
        return json.loads(out) if out else []

    def check_available(self) -> bool:
        """True when rclone can reach the curated bucket right now — one uncached liveness probe.

        Exists to tell "this one scene failed" apart from "rclone has stopped working", which no
        exception type or exit code can do: a dead remote surfaces as a bare RuntimeError out of
        _run_streaming, exactly like a reconstruction error, and rclone check returns exit 1 for a
        content difference and an unreachable remote alike. So the question is asked directly.

        Deliberately bypasses _cached: the 60s TTL would otherwise hand back a success recorded
        before the credentials expired and defeat the entire point of asking. An empty bucket is
        still *available* — this is liveness, not content.
        """
        # __init__ degrades to a None client when rclone was misconfigured at construction, and
        # nothing else ever retries. Without this the probe would answer with a construction-time
        # verdict for the life of the process — a lie for a method documented to ask about *now*,
        # and a long-lived dashboard session started during a transient fault could never recover.
        if self._client is None:
            try:
                self._client = RcloneClient()
            except Exception as exc:
                logger.error("rclone still unavailable on re-attempt: %s", exc)
                return False
        try:
            self._lsjson(CURATED_BUCKET, "")
        except Exception as exc:
            # Catch-all on purpose: a probe whose job is to answer "is rclone working?" must return
            # an answer, never raise the very fault it was asked to detect.
            logger.error("rclone liveness probe failed against %s: %s", CURATED_BUCKET, exc)
            return False
        return True

    ########
    # Curated inputs
    ########

    # Both curated listings go through _lsjson for the same reason the processed probes do: an
    # unreachable bucket must not read as an empty one. run_pipeline_remote takes its entire work
    # list from list_scenes(), so a [] from a broken rclone made it log "No scenes to process" and
    # exit 2 — an unattended driver quietly reporting that a full bucket had nothing in it.

    def list_scenes(self) -> list[str]:
        """Curated scene ids — dir names directly under environments-curated."""

        def _produce():
            entries = self._lsjson(CURATED_BUCKET, "")
            if not entries:
                logger.info("%s listed empty — no curated scenes to process", CURATED_BUCKET)
                return []
            dirs = [e["Name"] for e in entries if e.get("IsDir")]
            # A present-but-unmatched dir is exactly how a naming-convention drift looks; name it,
            # or the next batch of curated scenes silently vanishes from --all like audiomoth did.
            skipped = sorted(d for d in dirs if not SCENE_ID_RE.match(d))
            if skipped:
                logger.warning("skipping non-scene dirs in %s: %s", CURATED_BUCKET, ", ".join(skipped))
            return sorted(d for d in dirs if SCENE_ID_RE.match(d))

        return self._cached(("list_scenes",), _produce)

    def scene_video(self, scene: str) -> str:
        """The single video filename inside a curated scene dir."""

        def _produce():
            entries = self._lsjson(CURATED_BUCKET, scene)
            videos = sorted(
                e["Name"] for e in entries if not e.get("IsDir") and e["Name"].lower().endswith(_VIDEO_EXTS)
            )
            # Reported here, inside the producer, so it fires once per TTL rather than on every
            # cached hit — the dashboard probes the same scene repeatedly.
            if not videos:
                logger.info("no video in %s/%s — skipping", CURATED_BUCKET, scene)
            return videos

        # Raising outside the producer memoizes only the listing. A scene dir that lists but holds no
        # video is a stable fact, so it caches [] and re-raises FileNotFoundError from cache for the
        # TTL. A fault now raises out of _lsjson *inside* the producer, so it never reaches the memo
        # and the next call retries — previously list_directory swallowed it into [], which memoized
        # and reported an unreachable bucket as "this scene has no video" for the whole TTL.
        videos = self._cached(("scene_video", scene), _produce)
        if not videos:
            raise FileNotFoundError(f"no video in {CURATED_BUCKET}/{scene}")
        if len(videos) > 1:
            logger.warning("%s holds %d videos; using %s", scene, len(videos), videos[0])
        return videos[0]

    def fetch_video(self, scene: str, dest_dir: Path, on_line: Callable[[str], None] | None = None) -> Path:
        """Copy the scene's video to dest_dir; returns the local path."""
        name = self.scene_video(scene)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        cmd = self._require_client()._cmd(
            "copyto",
            f"{self._remote}:{CURATED_BUCKET}/{scene}/{name}",
            str(local),
            *_stats_args("2s"),
        )
        self._run_streaming(cmd, f"copyto to {local}", on_line=on_line)
        return local

    ########
    # Processed state
    ########

    # Each probe below answers "is this already there?", so it must separate *absent* from
    # *unreachable*: an absent path is a genuine answer (memoized like any other), while a broken
    # rclone propagates so the caller aborts loudly instead of treating the whole bucket as
    # unprocessed. A raised producer never reaches the memo, so a blip cannot poison the TTL.
    # Absence arrives as an empty listing, not as an exit code — see _lsjson — so the empty branch
    # is where the skip-and-report log belongs.

    def list_localization_dbs(self, scene: str) -> list[str]:
        """Extractor names with a feature DB in the remote zarr (memoized directory listing)."""

        def _produce():
            path = f"{scene}/pointcloud.zarr/local_features"
            entries = self._lsjson(PROCESSED_BUCKET, path)
            if not entries:
                logger.info("no localization DBs for %s: %s/%s listed empty", scene, PROCESSED_BUCKET, path)
                return []
            return sorted(e["Name"] for e in entries if e.get("IsDir"))

        return self._cached(("list_localization_dbs", scene), _produce)

    def list_processed_scenes(self) -> list[str]:
        """Scene ids with processed outputs — dir names directly under environments-processed."""

        # SCENE_ID_RE filters here too: every processed dir is created by our own push_outputs, so a
        # non-matching name is a stray upload rather than a scene.
        def _produce():
            entries = self._lsjson(PROCESSED_BUCKET, "")
            if not entries:
                logger.info("%s listed empty — nothing processed yet", PROCESSED_BUCKET)
                return []
            return sorted(e["Name"] for e in entries if e.get("IsDir") and SCENE_ID_RE.match(e["Name"]))

        return self._cached(("list_processed_scenes",), _produce)

    def has_processed(self, scene: str) -> bool:
        """True if processed outputs already exist for this scene (memoized)."""

        def _produce():
            entries = self._lsjson(PROCESSED_BUCKET, scene)
            if not entries:
                logger.info("%s has no processed outputs yet — %s listed empty", scene, PROCESSED_BUCKET)
                return False
            return True

        return self._cached(("has_processed", scene), _produce)

    ########
    # Transfers
    ########

    def pull_processed(
        self,
        scene: str,
        dest_dir: Path,
        excludes: tuple = (),
        on_line: Callable[[str], None] | None = None,
    ) -> Path:
        """rclone-copy processed outputs to dest_dir; return the local dir.

        excludes: rclone --exclude patterns (e.g. "/images/**") to skip artifacts a
        consumer does not need. on_line, if given, receives each --stats progress line.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{self._remote}:{PROCESSED_BUCKET}/{scene}"
        # Verb + endpoints first, then caller excludes and one-line stats.
        args = ["copy", remote, str(dest_dir)]
        for pattern in excludes:
            args += ["--exclude", pattern]
        args += _stats_args("2s")
        self._run_streaming(self._require_client()._cmd(*args), f"copy from {remote}", on_line=on_line)
        return dest_dir

    def pull_zarr_members(
        self,
        scene: str,
        dest_dir: Path,
        members: tuple,
        on_line: Callable[[str], None] | None = None,
    ) -> None:
        """Fetch named pointcloud.zarr member arrays (names relative to the zarr root).

        The scene pull skips the dense per-pixel arrays (PULL_EXCLUDES) because display
        never needs them — but the feature lift on legacy scenes (no cached
        semantics/<extractor>_lifted.zarr) does. This pulls exactly the named members on demand.
        """
        dest = Path(dest_dir) / "pointcloud.zarr"
        dest.mkdir(parents=True, exist_ok=True)
        remote = f"{self._remote}:{PROCESSED_BUCKET}/{scene}/pointcloud.zarr"
        args = ["copy", remote, str(dest)]
        for member in members:
            args += ["--include", f"{member}/**"]
        args += _stats_args("2s")
        self._run_streaming(self._require_client()._cmd(*args), f"copy members from {remote}", on_line=on_line)

    def push_outputs(self, local_dir: Path, scene: str, on_line: Callable[[str], None] | None = None) -> None:
        """Upload a processed scene dir to environments-processed/<scene>/."""
        args = [
            "copy",
            str(local_dir),
            f"{self._remote}:{PROCESSED_BUCKET}/{scene}",
            "--gcs-bucket-policy-only",
            "--transfers",
            "8",
            "--retries",
            "3",
            "--timeout",
            "300s",
            "--contimeout",
            "60s",
            *_stats_args("2s"),
        ]
        # Regenerable artifacts stay local — see PUSH_EXCLUDES
        for pattern in PUSH_EXCLUDES:
            args += ["--exclude", pattern]
        self._run_streaming(self._require_client()._cmd(*args), "push", on_line)

        # Anything cached about this scene's processed state is now stale
        self.invalidate(("has_processed", scene))
        self.invalidate(("list_localization_dbs", scene))
        self.invalidate(("list_processed_scenes",))

    def verify_push(self, local_dir: Path, scene: str, on_line: Callable[[str], None] | None = None) -> bool:
        """True when every local file (minus PUSH_EXCLUDES) exists remotely and matches.

        --one-way so remote-only leftovers from an earlier run are not a failure. This is
        the gate the remote driver requires before deleting anything local.
        """
        # Everything, argv included, runs inside the one try: `self._remote` and _require_client()
        # both raise RuntimeError on a missing/misconfigured rclone and both land in the non-marker
        # branch below, whose message names the cause. Either way the answer is False — callers gate
        # a destructive local delete on True, so "cannot verify" must never look like "verified".
        try:
            remote = f"{self._remote}:{PROCESSED_BUCKET}/{scene}"
            args = [
                "check",
                str(local_dir),
                remote,
                "--one-way",
                "--gcs-bucket-policy-only",
                # images/ is pushed, so this walks one PNG per keyframe: list
                # recursively (one request per prefix instead of one per directory) and widen the
                # checker pool. Deliberately NO --size-only/--checksum downgrade — this gates a
                # destructive local delete, so it must stay a full content comparison.
                "--fast-list",
                "--checkers",
                "16",
                # A check reports no bytes, so these lines are a liveness heartbeat rather than
                # progress; 15s keeps a long verify visibly alive without flooding the log.
                *_stats_args("15s"),
            ]
            # Excluded files were never uploaded — checking them would always fail
            for pattern in PUSH_EXCLUDES:
                args += ["--exclude", pattern]
            # rclone's check stats carry no byte counters, so the interval lines are pure liveness
            # with no scene in them — announce what is being verified before they start.
            logger.info("verifying %s against %s (full content check, may take a while)", local_dir, remote)
            self._run_streaming(self._require_client()._cmd(*args), "check", on_line)
        except (RuntimeError, OSError) as exc:
            # Fail-safe either way: False blocks the delete. But the *cause* must be right — logging
            # an unreachable bucket as a mismatch sends an operator hunting a corrupted upload.
            if _CHECK_MISMATCH_MARKER in str(exc):
                logger.error("verify FAILED for %s — content mismatch, local data kept: %s", scene, exc)
            else:
                logger.error("verify FAILED for %s — check could not run, local data kept: %s", scene, exc)
            return False
        return True
