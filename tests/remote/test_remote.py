"""SceneSource: flat scene ids, injectable buckets, verified push, caching."""

import fnmatch
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from collab_data.data_dashboard.rclone_client import RcloneClient

import collab_splats.remote as remote
from collab_splats.remote import PUSH_EXCLUDES, SCENE_ID_RE, SceneSource

########
# Video-name case matrix
########

# Every case form of each video extension the push must exclude, generated so no bracket class is missed
_VIDEO_EXT_STEMS = ("mp4", "mov", "avi")


def _ext_case_variants(stem: str) -> list[str]:
    """lowercase, UPPERCASE, MixedCase, and every single-letter case flip of an extension."""
    variants = {stem.lower(), stem.upper(), stem.capitalize()}
    variants.update(stem[:i] + ch.upper() + stem[i + 1 :] for i, ch in enumerate(stem) if ch.isalpha())
    return sorted(variants)


_VIDEO_NAME_CASES = [f"C0043.{v}" for stem in _VIDEO_EXT_STEMS for v in _ext_case_variants(stem)]


class FakeClient:
    """Records every transport call; returns canned listings."""

    remote_name = "fake"

    def __init__(self, listings=None, fail=None):
        self.listings = listings or {}
        self.fail = fail
        self.calls = []

    def _raise(self):
        if isinstance(self.fail, BaseException):
            raise self.fail
        raise RuntimeError(self.fail)

    def run(self, *args):
        self.calls.append(("run", args))
        if self.fail:
            self._raise()
        return json.dumps(self.listings.get(args[-1], []))

    def run_streaming(self, *args, on_line=None):
        self.calls.append(("run_streaming", args))

    def copy_dir(self, src, dst, exclude=(), on_line=None, extra=()):
        self.calls.append(("copy_dir", src, dst, tuple(exclude), tuple(extra)))

    def check(self, src, dst, exclude=(), extra=()):
        self.calls.append(("check", src, dst, tuple(exclude)))
        if self.fail:
            self._raise()
        return True


def _dirs(*names):
    return [{"Name": n, "IsDir": True} for n in names]


def _files(*names):
    return [{"Name": n, "IsDir": False} for n in names]


def _runs(client):
    """The listing calls a client saw, transfers excluded."""
    return [call for call in client.calls if call[0] == "run"]


########
# Probe matrix — absent vs unreachable
########

# Memoized processed-state probes and their answer for an absent path (GCS: exit 0 + [])
_PROBES = {
    "has_processed": (lambda src: src.has_processed("s"), False),
    "list_localization_dbs": (lambda src: src.list_localization_dbs("s"), []),
    "list_processed_scenes": (lambda src: src.list_processed_scenes(), []),
}

# Curated listings have no absent answer: a fault must raise, never list as an empty bucket
_CURATED_CALLS = {
    "list_scenes": lambda src: src.list_scenes(),
    "scene_video": lambda src: src.scene_video("s"),
}

# Every non-zero exit is a fault, 3 and 4 included: on GCS exit 3 means a missing bucket
_TRANSPORT_EXIT_CODES = (1, 2, 3, 4, 5, 7)


def _exit_error(code, detail="403 forbidden"):
    """The RuntimeError text RcloneClient.run raises for a non-zero exit."""
    return f"rclone lsjson failed (exit {code}): {detail}"


########
# Buckets + listing
########


def test_bucket_names_are_kwargs():
    client = FakeClient(listings={"fake:cur": [{"Name": "scene_a", "IsDir": True}]})
    source = SceneSource(client, curated="cur", processed="proc")
    assert source.list_scenes() == ["scene_a"]


def test_processed_bucket_kwarg_reaches_the_path():
    client = FakeClient(listings={"fake:proc": _dirs("scene_a")})
    source = SceneSource(client, curated="cur", processed="proc")
    assert source.list_processed_scenes() == ["scene_a"]


def test_default_buckets_are_the_environments_pair():
    source = SceneSource(FakeClient())
    assert source.curated == "environments-curated"
    assert source.processed == "environments-processed"


def test_list_scenes_returns_dirs_at_bucket_root():
    # ".cache" fails SCENE_ID_RE (leading dot); the scene-named .mp4 matches it but is not a dir
    listings = {
        "fake:environments-curated": _dirs("2026_07_21-rats-C0100", ".cache", "2026_07_20-birds-C0043")
        + _files("notes.txt", "2026_05_07-birds-clip_03.mp4")
    }
    source = SceneSource(FakeClient(listings=listings))
    assert source.list_scenes() == ["2026_07_20-birds-C0043", "2026_07_21-rats-C0100"]


def test_scene_id_re_is_path_safety_not_date_shape():
    """The regex guards the output-path join; the YYYY_MM_DD shape is convention, not contract."""
    accepted = [
        "2026_07_20-birds-C0043",
        "audiomoth_only_deployments-20260810_20260831-boston_frontageroad_tracks-splat_videos-GH010250",
        "audiomoth_only_deployments-20260817_20260824-boston_publicgarden_maintenance-splat_videos-GH010257",
    ]
    for name in accepted:
        assert SCENE_ID_RE.match(name), name

    # One path segment, no leading dot (kills "..", ".", hidden dirs), no leading "-" (argv-safe)
    rejected = ["../x", "..", ".", ".hidden", "a/b", "", "-flag"]
    for name in rejected:
        assert not SCENE_ID_RE.match(name), name


def test_scene_video_picks_the_single_video():
    listings = {"fake:environments-curated/s": _files("C0043.MP4", "readme.md")}
    assert SceneSource(FakeClient(listings=listings)).scene_video("s") == "C0043.MP4"


@pytest.mark.parametrize("name", ("C0043.MP4", "clip.mp4", "clip.MOV", "clip.mov", "clip.AVI", "clip.avi"))
def test_scene_video_accepts_every_video_extension(name):
    listings = {"fake:environments-curated/s": _files(name, "readme.md")}
    assert SceneSource(FakeClient(listings=listings)).scene_video("s") == name


def test_video_exts_kwarg_narrows_the_accepted_videos():
    listings = {"fake:environments-curated/s": _files("a.mp4", "b.mov")}
    source = SceneSource(FakeClient(listings=listings), video_exts=(".mov",))
    assert source.scene_video("s") == "b.mov"


def test_scene_video_takes_the_first_of_several_videos():
    listings = {"fake:environments-curated/s": _files("b_second.MP4", "a_first.mp4")}
    assert SceneSource(FakeClient(listings=listings)).scene_video("s") == "a_first.mp4"


def test_scene_video_caches_per_scene():
    listings = {
        "fake:environments-curated/a": _files("A.MP4"),
        "fake:environments-curated/b": _files("B.MP4"),
    }
    source = SceneSource(FakeClient(listings=listings))
    assert source.scene_video("a") == "A.MP4"
    assert source.scene_video("b") == "B.MP4"


def test_scene_video_raises_when_no_video(caplog):
    """A dir that lists but holds no video is a real answer — report it, do not treat it as a fault."""
    listings = {"fake:environments-curated/s": _files("readme.md")}
    with caplog.at_level(logging.INFO, logger="collab_splats.remote"):
        with pytest.raises(FileNotFoundError, match="no video"):
            SceneSource(FakeClient(listings=listings)).scene_video("s")
    assert caplog.records, "a video-less scene dir must be reported, not skipped in silence"


@pytest.mark.parametrize("call", sorted(_CURATED_CALLS))
@pytest.mark.parametrize("code", _TRANSPORT_EXIT_CODES)
def test_curated_listing_raises_on_rclone_failure(call, code):
    """A curated listing must never collapse a fault into [] — the driver would see an empty bucket."""
    source = SceneSource(FakeClient(fail=_exit_error(code)))
    with pytest.raises(RuntimeError, match="403 forbidden"):
        _CURATED_CALLS[call](source)


@pytest.mark.parametrize("call", sorted(_CURATED_CALLS))
def test_curated_listing_does_not_memoize_a_failure(call):
    """A fault must re-attempt on the next call."""
    client = FakeClient(fail=_exit_error(5, "temporary"))
    source = SceneSource(client)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            _CURATED_CALLS[call](source)
    assert len(_runs(client)) == 2
    assert not source._listing_cache


def test_list_scenes_logs_when_the_curated_bucket_is_empty(caplog):
    """An empty bucket is a genuine answer, but it must say so."""
    with caplog.at_level(logging.INFO, logger="collab_splats.remote"):
        assert SceneSource(FakeClient()).list_scenes() == []
    assert caplog.records, "an empty curated bucket must be logged"


def test_list_scenes_warns_about_skipped_non_scene_dirs(caplog):
    """A dir that fails SCENE_ID_RE must be named in the log, never dropped silently."""
    listings = {"fake:environments-curated": _dirs("2026_07_20-birds-C0043", ".cache")}
    with caplog.at_level(logging.WARNING, logger="collab_splats.remote"):
        assert SceneSource(FakeClient(listings=listings)).list_scenes() == ["2026_07_20-birds-C0043"]
    assert any(".cache" in r.getMessage() for r in caplog.records), "skipped dir must be logged"


def test_list_processed_scenes_reads_processed_bucket():
    # A non-scene dir and a scene-named file must both be dropped
    listings = {
        "fake:environments-processed": _dirs("2026_07_21-rats-C0100", ".cache", "2026_07_20-birds-C0043")
        + _files("2026_05_07-birds-clip_03.mp4")
    }
    source = SceneSource(FakeClient(listings=listings))
    assert source.list_processed_scenes() == ["2026_07_20-birds-C0043", "2026_07_21-rats-C0100"]


def test_has_processed_true_when_listing_nonempty():
    listings = {"fake:environments-processed/s": _files("transforms.json")}
    assert SceneSource(FakeClient(listings=listings)).has_processed("s") is True


def test_list_localization_dbs_returns_extractor_dirs():
    path = "fake:environments-processed/s/pointcloud.zarr/local_features"
    listings = {path: _dirs("xfeat", "disk") + _files("zarr.json")}
    assert SceneSource(FakeClient(listings=listings)).list_localization_dbs("s") == ["disk", "xfeat"]


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_reports_empty_and_logs_when_path_is_absent(caplog, probe):
    """An unprocessed scene lists as exit 0 with [] — the real "nothing processed yet" answer."""
    call, absent_answer = _PROBES[probe]
    with caplog.at_level(logging.INFO, logger="collab_splats.remote"):
        assert call(SceneSource(FakeClient())) == absent_answer
    assert caplog.records, f"{probe} must log the absent path"


@pytest.mark.parametrize("probe", sorted(_PROBES))
@pytest.mark.parametrize("code", _TRANSPORT_EXIT_CODES)
def test_probe_raises_on_any_nonzero_rclone_exit(probe, code):
    """A broken rclone must abort the caller, not report every scene as unprocessed."""
    call, _ = _PROBES[probe]
    with pytest.raises(RuntimeError, match="403 forbidden"):
        call(SceneSource(FakeClient(fail=_exit_error(code))))


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_raises_on_a_missing_bucket(probe):
    """Exit 3 means the bucket is gone or misspelled, so it is a fault and never an absence."""
    call, _ = _PROBES[probe]
    with pytest.raises(RuntimeError, match="directory not found"):
        call(SceneSource(FakeClient(fail=_exit_error(3, "directory not found"))))


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_raises_when_the_rclone_binary_is_missing(probe):
    call, _ = _PROBES[probe]
    client = FakeClient(fail=FileNotFoundError(2, "No such file or directory: 'rclone'"))
    with pytest.raises(OSError):
        call(SceneSource(client))


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_raises_when_the_client_is_unavailable(probe):
    """RcloneClient failed to construct: unknown, not empty."""
    call, _ = _PROBES[probe]
    source = SceneSource(FakeClient())
    source._client = None
    with pytest.raises(RuntimeError, match="not available"):
        call(source)


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_does_not_memoize_a_transport_failure(probe):
    """One blip must not poison an entire sweep for the TTL."""
    call, _ = _PROBES[probe]
    client = FakeClient(fail=_exit_error(5, "temporary"))
    source = SceneSource(client)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            call(source)
    assert len(_runs(client)) == 2, "a failed probe must re-attempt, not serve a cached failure"
    assert not source._listing_cache


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_memoizes_a_genuinely_absent_path(probe):
    """Absent is a fact about the bucket, so it memoizes like any other answer."""
    call, absent_answer = _PROBES[probe]
    client = FakeClient()
    source = SceneSource(client)
    assert call(source) == absent_answer
    assert call(source) == absent_answer
    assert len(_runs(client)) == 1


def test_listing_ttl_kwarg_expires_the_memo():
    client = FakeClient(listings={"fake:environments-curated": _dirs("a")})
    source = SceneSource(client, listing_ttl=0.0)
    source.list_scenes()
    source.list_scenes()
    assert len(_runs(client)) == 2


def test_scene_video_reports_a_missing_video_once_per_ttl(caplog):
    """The report lives inside the producer, so a cached call does not re-fire it."""
    source = SceneSource(FakeClient(listings={"fake:environments-curated/s": _files("readme.md")}))
    with caplog.at_level(logging.INFO, logger="collab_splats.remote"):
        for _ in range(3):
            with pytest.raises(FileNotFoundError):
                source.scene_video("s")
    assert len([r for r in caplog.records if "no video" in r.getMessage()]) == 1


########
# Liveness — "did this one scene fail, or has rclone stopped working?"
########


def test_check_available_true_when_the_curated_bucket_lists():
    client = FakeClient(listings={"fake:environments-curated": _dirs("2026_07_20-birds-C0043")})
    assert SceneSource(client).check_available() is True


def test_check_available_true_on_an_empty_bucket():
    """Liveness, not content: an empty bucket is a reachable bucket."""
    assert SceneSource(FakeClient()).check_available() is True


@pytest.mark.parametrize("code", _TRANSPORT_EXIT_CODES)
def test_check_available_false_on_any_rclone_failure(code):
    assert SceneSource(FakeClient(fail=_exit_error(code, "403"))).check_available() is False


def test_check_available_false_when_the_rclone_binary_is_missing():
    """The probe answers a question; it must never raise the very fault it was asked about."""
    client = FakeClient(fail=FileNotFoundError(2, "No such file or directory: 'rclone'"))
    assert SceneSource(client).check_available() is False


def test_check_available_false_when_the_client_cannot_be_constructed(monkeypatch):
    def _boom():
        raise RuntimeError("rclone is not available")

    monkeypatch.setattr(remote, "RcloneClient", _boom)
    source = SceneSource(FakeClient())
    source._client = None
    assert source.check_available() is False


def test_check_available_reattempts_client_construction(monkeypatch):
    """A source born during a transient rclone fault must recover once rclone is healthy."""
    source = SceneSource(FakeClient())
    source._client = None
    recovered = FakeClient()
    monkeypatch.setattr(remote, "RcloneClient", lambda: recovered)
    assert source.check_available() is True
    assert source._client is recovered


def test_check_available_bypasses_the_listing_memo():
    """A success memoized before the credentials expired must not answer about now."""
    client = FakeClient(listings={"fake:environments-curated": _dirs("2026_07_20-birds-C0043")})
    source = SceneSource(client)
    source.list_scenes()
    assert source.check_available() is True
    baseline = len(_runs(client))

    # The remote dies while the memo still holds the earlier success
    client.fail = _exit_error(5, "temporary")
    assert ("list_scenes",) in source._listing_cache
    assert source.check_available() is False
    assert len(_runs(client)) == baseline + 1, "the probe must re-run rclone, not read the memo"


def test_check_available_does_not_populate_the_memo():
    client = FakeClient(listings={"fake:environments-curated": _dirs("2026_07_20-birds-C0043")})
    source = SceneSource(client)
    source.check_available()
    assert source._listing_cache == {}


########
# Transfers — one scene id, no reconstruction/ prefix
########


def test_fetch_video_copies_to_dest(tmp_path):
    client = FakeClient(listings={"fake:environments-curated/s": _files("C0043.MP4")})
    local = SceneSource(client).fetch_video("s", dest_dir=tmp_path)
    assert local == tmp_path / "C0043.MP4"

    kind, args = client.calls[-1]
    assert kind == "run_streaming"
    assert args[:3] == ("copyto", "fake:environments-curated/s/C0043.MP4", str(local))
    assert "--stats-one-line" in args


def test_push_outputs_targets_scene_dir_with_excludes(tmp_path):
    client = FakeClient()
    SceneSource(client).push_outputs(tmp_path, "s")
    _, src, dst, exclude, extra = client.calls[0]
    assert (src, dst) == (str(tmp_path), "fake:environments-processed/s")
    assert exclude == PUSH_EXCLUDES
    assert "--gcs-bucket-policy-only" in extra


def test_push_excludes_only_the_temporary_full_width_features():
    """
    The temporary full-width features stay local; the 2D codes and the lifted stores are pushed.

    - rclone reads a leading slash as "relative to the transfer root"; strip it to model that
    """
    patterns = [p.lstrip("/") for p in PUSH_EXCLUDES]
    assert "/semantics/*_features.zarr/**" in PUSH_EXCLUDES
    assert any(fnmatch.fnmatchcase("semantics/dinov2_features.zarr/features/c/0/0/0/0", p) for p in patterns)

    # Codes store, its AE and the backend's lifted store all travel
    kept = (
        "semantics/dinov2_codes.zarr/features/c/0/0/0/0",
        "semantics/dinov2_codes.zarr/autoencoder.pt",
        "vggt_omega/semantics/dinov2_lifted.zarr/features/c/0/0",
    )

    for name in kept:
        assert not any(fnmatch.fnmatchcase(name, p) for p in patterns), name


@pytest.mark.parametrize("name", ("images", "images/frame_000000.png", "images/frame_000123.png"))
def test_push_keeps_the_keyframe_images(name):
    """The scene-root images/ store is the sole persistent frame source, so a pull must carry it."""
    assert not any(fnmatch.fnmatchcase(name, p.lstrip("/")) for p in PUSH_EXCLUDES), name
    assert "/images/**" not in PUSH_EXCLUDES

    # Unanchored it would also swallow pointcloud.zarr/images and every <backend>/images
    assert "images/**" not in PUSH_EXCLUDES


@pytest.mark.parametrize(
    "name",
    (
        "colmap/colmap/colmap.db",
        "instantsfm/colmap/instantsfm.db",
        "hloc/colmap/hloc/feats-superpoint-n4096-rmax1600.h5",
        "hloc/colmap/hloc/sfm/database.db",
    ),
)
def test_push_excludes_cover_the_colmap_and_hloc_build_artifacts(name):
    assert any(fnmatch.fnmatchcase(name, p.lstrip("/")) for p in PUSH_EXCLUDES), name


def test_push_excludes_keep_the_colmap_model():
    """The model the dense result was built on must reach processed."""
    assert not any(fnmatch.fnmatchcase("hloc/colmap/sparse/0/images.bin", p.lstrip("/")) for p in PUSH_EXCLUDES)


@pytest.mark.parametrize("name", _VIDEO_NAME_CASES)
def test_push_excludes_the_source_video_in_any_case(name):
    """The remote driver fetches the video into the dir it pushes, so it must be excluded."""
    assert any(fnmatch.fnmatchcase(name, p) for p in PUSH_EXCLUDES), name


@pytest.mark.parametrize("name", ("sparse_pc.ply", "run_config.yaml", "mesh.obj"))
def test_push_keeps_non_video_outputs(name):
    assert not any(fnmatch.fnmatchcase(name, p) for p in PUSH_EXCLUDES), name


def test_pull_processed_copies_and_passes_exclude_flags(tmp_path):
    client = FakeClient()
    SceneSource(client).pull_processed("s", tmp_path, excludes=("/images/**",))
    _, src, dst, exclude, _ = client.calls[0]
    assert (src, dst) == ("fake:environments-processed/s", str(tmp_path))
    assert exclude == ("/images/**",)


def test_pull_zarr_members_includes_each_bare_member(tmp_path):
    """Members are zarr-root-relative bare names, pulled as `<name>/**`."""
    client = FakeClient()
    SceneSource(client).pull_zarr_members("s", tmp_path, members=("pixel_indices", "depth"))
    kind, args = client.calls[0]
    assert kind == "run_streaming"

    # The remote is already rooted at pointcloud.zarr, so an --include may not repeat that prefix
    assert args[:3] == ("copy", "fake:environments-processed/s/pointcloud.zarr", str(tmp_path / "pointcloud.zarr"))
    includes = [args[i + 1] for i, arg in enumerate(args) if arg == "--include"]
    assert includes == ["pixel_indices/**", "depth/**"]


########
# Verify
########


def test_verify_push_true_on_clean_check(tmp_path):
    client = FakeClient()
    assert SceneSource(client).verify_push(tmp_path, "s") is True

    # Operand ORDER is load-bearing: local first asks "did everything I have get uploaded?"
    _, src, dst, exclude = client.calls[0]
    assert (src, dst) == (str(tmp_path), "fake:environments-processed/s")
    assert exclude == PUSH_EXCLUDES


def test_verify_push_false_on_mismatch(tmp_path):
    client = FakeClient()
    client.check = lambda *a, **k: False
    assert SceneSource(client).verify_push(tmp_path, "s") is False


def test_verify_push_logs_a_content_mismatch(tmp_path, caplog):
    client = FakeClient()
    client.check = lambda *a, **k: False
    with caplog.at_level(logging.ERROR, logger="collab_splats.remote"):
        assert SceneSource(client).verify_push(tmp_path, "s") is False
    assert "content mismatch" in " ".join(r.getMessage() for r in caplog.records)


def test_verify_push_answers_false_when_the_check_cannot_run(tmp_path):
    source = SceneSource(FakeClient(fail="boom"))
    assert source.verify_push(tmp_path, "s") is False


def test_verify_push_logs_a_check_that_could_not_run(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="collab_splats.remote"):
        SceneSource(FakeClient(fail="boom")).verify_push(tmp_path, "s")
    assert "could not run" in " ".join(r.getMessage() for r in caplog.records).lower()


def test_verify_push_false_when_the_rclone_binary_is_missing(tmp_path):
    client = FakeClient(fail=FileNotFoundError(2, "No such file or directory: 'rclone'"))
    assert SceneSource(client).verify_push(tmp_path, "s") is False


def test_verify_push_false_when_rclone_unavailable(monkeypatch):
    """Cannot-verify must never read as verified — callers gate a destructive local delete on True."""

    def _boom():
        raise RuntimeError("rclone is not available")

    monkeypatch.setattr(remote, "RcloneClient", _boom)
    source = SceneSource()
    assert source._client is None
    assert source.verify_push(Path("/nonexistent/local"), "s") is False


def test_verify_push_keeps_the_content_comparison(tmp_path):
    """The check gates a destructive delete, so it must never downgrade to size-only."""
    calls = []

    class _Client(FakeClient):
        def check(self, src, dst, exclude=(), extra=()):
            calls.append(list(extra))
            return True

    SceneSource(_Client()).verify_push(tmp_path, "s")
    assert "--fast-list" in calls[0]
    for weakening in ("--size-only", "--ignore-checksum", "--checksum"):
        assert weakening not in calls[0]


########
# Push excludes — anchoring, so a pattern for a root dir cannot swallow a nested namesake
########


def test_push_excludes_anchor_the_2d_features_at_the_scene_root():
    """Unanchored, the features pattern could also exclude `<backend>/semantics/**` — the deliverable."""
    assert "/semantics/*_features.zarr/**" in PUSH_EXCLUDES
    assert "semantics/*_features.zarr/**" not in PUSH_EXCLUDES, "unanchored: rclone matches it at any depth"

    # Nothing may exclude the lifted pair under the backend dir, at any depth
    assert not any("semantics" in p and not p.startswith("/") for p in PUSH_EXCLUDES)


########
# Cache + misc
########


def test_listing_cache_hits_once():
    client = FakeClient(listings={"fake:environments-curated": _dirs("2026_07_20-birds-C0043")})
    source = SceneSource(client)
    source.list_scenes()
    source.list_scenes()
    assert len(_runs(client)) == 1


def test_invalidate_drops_one_key():
    client = FakeClient(
        listings={
            "fake:environments-processed": _dirs("2026_07_20-birds-C0043"),
            "fake:environments-curated": _dirs("2026_07_20-birds-C0043"),
        }
    )
    source = SceneSource(client)
    source.list_scenes()
    source.list_processed_scenes()
    assert len(_runs(client)) == 2

    source.invalidate(("list_scenes",))
    source.list_processed_scenes()
    assert len(_runs(client)) == 2, "still cached"

    source.list_scenes()
    assert len(_runs(client)) == 3, "dropped, refetches"


def test_push_outputs_invalidates_processed_listings(tmp_path):
    client = FakeClient(
        listings={
            "fake:environments-processed/s": _files("transforms.json"),
            "fake:environments-processed": _dirs("2026_07_20-birds-C0043"),
            "fake:environments-processed/s/pointcloud.zarr/local_features": _dirs("disk"),
        }
    )
    source = SceneSource(client)
    source.has_processed("s")
    source.list_localization_dbs("s")
    source.list_processed_scenes()

    # Each producer key must be cached before the push, so a renamed key fails here
    keys = [("has_processed", "s"), ("list_localization_dbs", "s"), ("list_processed_scenes",)]
    for key in keys:
        assert key in source._listing_cache

    source.push_outputs(tmp_path, "s")
    for key in keys:
        assert key not in source._listing_cache


########
# Real rclone against a local remote
########


@pytest.fixture()
def local_source(monkeypatch, tmp_path):
    """SceneSource over rclone's local backend, buckets are tmp dirs."""
    if shutil.which("rclone") is None:
        pytest.skip("rclone not installed")
    monkeypatch.setenv("RCLONE_CONFIG_LT_TYPE", "local")
    curated = tmp_path / "curated"
    processed = tmp_path / "processed"
    curated.mkdir()
    processed.mkdir()
    return SceneSource(RcloneClient(remote_name="lt"), curated=str(curated), processed=str(processed))


def test_local_remote_lists_and_fetches_a_scene_video(local_source, tmp_path):
    scene_dir = Path(local_source.curated) / "scene_a"
    scene_dir.mkdir()
    (scene_dir / "clip.mp4").write_bytes(b"video")

    assert local_source.list_scenes() == ["scene_a"]

    dest = tmp_path / "fetched"
    assert local_source.fetch_video("scene_a", dest).read_bytes() == b"video"


def test_local_remote_push_verify_pull_round_trip(local_source, tmp_path):
    scene = tmp_path / "work" / "scene_a"
    scene.mkdir(parents=True)
    (scene / "mesh.ply").write_bytes(b"mesh")
    (scene / "clip.mp4").write_bytes(b"video")

    local_source.push_outputs(scene, "scene_a")
    pushed = Path(local_source.processed) / "scene_a"
    assert (pushed / "mesh.ply").read_bytes() == b"mesh"
    assert not (pushed / "clip.mp4").exists(), "source video is in PUSH_EXCLUDES"
    assert local_source.verify_push(scene, "scene_a") is True

    # A tampered remote file is a mismatch, not a fault
    (pushed / "mesh.ply").write_bytes(b"MESH")
    assert local_source.verify_push(scene, "scene_a") is False

    pulled = local_source.pull_processed("scene_a", tmp_path / "pulled")
    assert (pulled / "mesh.ply").read_bytes() == b"MESH"


def test_local_remote_missing_bucket_raises(local_source, tmp_path):
    source = SceneSource(RcloneClient(remote_name="lt"), curated=str(tmp_path / "nope"), processed=str(tmp_path))

    with pytest.raises(RuntimeError, match="exit 3"):
        source.list_scenes()


def test_importing_the_remote_module_stays_light():
    # The dashboard's fast bind imports collab_splats.remote, so it must never pull in torch
    code = (
        "import sys; import collab_splats.remote; "
        "assert 'torch' not in sys.modules, 'collab_splats.remote import pulled torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
