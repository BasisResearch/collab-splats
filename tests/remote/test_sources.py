"""SceneSource: flat scene ids, new buckets, verified push, caching."""

import fnmatch
import inspect
import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from collab_data.data_dashboard.rclone_client import RcloneClient

import collab_splats.remote.sources as sources
from collab_splats.remote.sources import (
    CURATED_BUCKET,
    PROCESSED_BUCKET,
    PULL_EXCLUDES,
    PUSH_EXCLUDES,
    SceneSource,
    parse_rclone_percent,
)

########
# Video-name case matrix
########

# Every video extension the push must exclude, in every case form. Generated rather than
# hand-listed so a regression in one bracket class (e.g. `*.[Aa][Vv][Ii]` -> `*.avi`) cannot
# slip through because nobody happened to type that filename.
_VIDEO_EXT_STEMS = ("mp4", "mov", "avi")


def _ext_case_variants(stem: str) -> list[str]:
    """lowercase, UPPERCASE, MixedCase, and every single-letter case flip of an extension."""
    variants = {stem.lower(), stem.upper(), stem.capitalize()}
    variants.update(stem[:i] + ch.upper() + stem[i + 1 :] for i, ch in enumerate(stem) if ch.isalpha())
    return sorted(variants)


_VIDEO_NAME_CASES = [f"C0043.{v}" for stem in _VIDEO_EXT_STEMS for v in _ext_case_variants(stem)]


class _FakeClient:
    """Stands in for RcloneClient: records argv, builds it exactly as the real _cmd does."""

    remote_name = "collab-data"

    def __init__(self, extra_args=()):
        # The real client carries per-remote flags here and _cmd appends them AFTER the operands.
        self.extra_args = list(extra_args)
        self.cmds = []

    def _cmd(self, *args):
        self.cmds.append(list(args))
        # Argv order mirrors RcloneClient._cmd — extra_args last, never before the path. Nothing is
        # executed: subprocess.run/Popen are monkeypatched in every test that reaches them.
        return ["rclone", *args, *self.extra_args]


def _dirs(*names):
    return [{"Name": n, "IsDir": True} for n in names]


def _files(*names):
    return [{"Name": n, "IsDir": False} for n in names]


def _remote_arg(cmd):
    """The `<remote>:<bucket>/<path>` operand, located by prefix rather than by position.

    Keying on cmd[-1] is what F4 was about: the real _cmd appends extra_args after the operands,
    so the remote path is only last when extra_args happens to be empty.
    """
    remotes = [a for a in cmd if a.startswith(f"{_FakeClient.remote_name}:")]
    assert len(remotes) == 1, f"expected exactly one remote operand in {cmd}"
    return remotes[0]


def _fake_popen(monkeypatch, lines=(), returncode=0, capture=None):
    """Replace subprocess.Popen so streaming helpers run without rclone."""

    class _P:
        def __init__(self, cmd, *a, **k):
            if capture is not None:
                capture.append(cmd)
            self.stdout = iter(lines)
            self.returncode = returncode

        def wait(self):
            return self.returncode

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(sources.subprocess, "Popen", _P)


def _fake_run(monkeypatch, listings=None, error=None):
    """Replace subprocess.run so the lsjson listings resolve without rclone.

    listings maps a full remote path ("collab-data:environments-processed/s") to its entries. A
    path missing from the map lists as exit 0 with an empty array, because that is precisely how
    GCS reports an absent prefix — prefixes are not directories, so there is nothing to 404 on.
    error, if given, is raised instead, standing in for a transport/config fault.
    """
    listings = listings or {}

    def _run(cmd, *a, **k):
        if error is not None:
            raise error
        entries = listings.get(_remote_arg(cmd), [])
        return SimpleNamespace(stdout=json.dumps(entries), stderr="", returncode=0)

    monkeypatch.setattr(sources.subprocess, "run", _run)


########
# Probe matrix — absent vs unreachable
########

# The memoized processed-state probes, each with the answer it must give for a genuinely absent
# path. On GCS "absent" arrives as exit 0 + [], never as a non-zero exit.
_PROBES = {
    "has_processed": (lambda src: src.has_processed("s"), False),
    "list_localization_dbs": (lambda src: src.list_localization_dbs("s"), []),
    "list_processed_scenes": (lambda src: src.list_processed_scenes(), []),
}

# The curated-side listings. These have no "absent" answer at all: both buckets are fixed module
# constants, so anything that stops a listing from resolving is a fault, and returning [] would
# tell the unattended driver the bucket is empty and let it exit 2 as if there were no work.
_CURATED_CALLS = {
    "list_scenes": lambda src: src.list_scenes(),
    "scene_video": lambda src: src.scene_video("s"),
}

# EVERY non-zero rclone exit is a fault now, 3 and 4 included. On a GCS backend a missing prefix
# does not 404 — it lists empty — so exit 3 means the *bucket* is missing or misspelled, which for
# a fixed constant is a config fault worth aborting on rather than "nothing processed yet".
_TRANSPORT_EXIT_CODES = (1, 2, 3, 4, 5, 7)


########
# Buckets + listing
########


def test_buckets_are_the_environments_pair():
    assert CURATED_BUCKET == "environments-curated"
    assert PROCESSED_BUCKET == "environments-processed"


def test_list_scenes_returns_dirs_at_bucket_root(monkeypatch):
    # Negative fixtures isolate each filter: "tmp" is a dir that fails SCENE_ID_RE, and the
    # scene-named .mp4 matches SCENE_ID_RE but is not a dir. Unsorted input pins the sort.
    _fake_run(
        monkeypatch,
        listings={
            f"collab-data:{CURATED_BUCKET}": _dirs("2026_07_21-rats-C0100", "tmp", "2026_07_20-birds-C0043")
            + _files("notes.txt", "2026_05_07-birds-clip_03.mp4")
        },
    )
    assert SceneSource(_FakeClient()).list_scenes() == ["2026_07_20-birds-C0043", "2026_07_21-rats-C0100"]


def test_scene_video_picks_the_single_video(monkeypatch):
    _fake_run(
        monkeypatch,
        listings={f"collab-data:{CURATED_BUCKET}/2026_07_20-birds-C0043": _files("C0043.MP4", "readme.md")},
    )
    assert SceneSource(_FakeClient()).scene_video("2026_07_20-birds-C0043") == "C0043.MP4"


@pytest.mark.parametrize("name", ("C0043.MP4", "clip.mp4", "clip.MOV", "clip.mov", "clip.AVI", "clip.avi"))
def test_scene_video_accepts_every_video_extension(monkeypatch, name):
    """.avi counts as a video: the local driver's batch.VIDEO_EXTS always took it, so a curated
    .avi that raised FileNotFoundError here made the remote driver skip the scene in silence."""
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}/s": _files(name, "readme.md")})
    assert SceneSource(_FakeClient()).scene_video("s") == name


def test_scene_video_takes_the_first_of_several_videos(monkeypatch):
    """A multi-video dir resolves to videos[0] of the sorted list, matching the logged warning."""
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}/s": _files("b_second.MP4", "a_first.mp4")})
    assert SceneSource(_FakeClient()).scene_video("s") == "a_first.mp4"


def test_scene_video_caches_per_scene(monkeypatch):
    """The memo key carries the scene, so a second scene must not read the first scene's video."""
    _fake_run(
        monkeypatch,
        listings={
            f"collab-data:{CURATED_BUCKET}/a": _files("A.MP4"),
            f"collab-data:{CURATED_BUCKET}/b": _files("B.MP4"),
        },
    )
    src = SceneSource(_FakeClient())
    assert src.scene_video("a") == "A.MP4"
    assert src.scene_video("b") == "B.MP4"


def test_scene_video_raises_when_no_video(monkeypatch, caplog):
    """A dir that lists but holds no video is a real answer — report it, do not treat it as a fault."""
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}/s": _files("readme.md")})
    with caplog.at_level(logging.INFO, logger="collab_splats.remote.sources"):
        with pytest.raises(FileNotFoundError, match="no video"):
            SceneSource(_FakeClient()).scene_video("s")
    assert caplog.records, "a video-less scene dir must be reported, not skipped in silence"


@pytest.mark.parametrize("call", sorted(_CURATED_CALLS))
@pytest.mark.parametrize("code", _TRANSPORT_EXIT_CODES)
def test_curated_listing_raises_on_rclone_failure(monkeypatch, call, code):
    """A curated listing must never collapse a fault into []. run_pipeline_remote takes its whole
    work list from list_scenes(), so [] makes it log "No scenes to process" and exit 2 — an
    unattended driver reporting an empty bucket when rclone is simply broken."""
    _fake_run(monkeypatch, error=subprocess.CalledProcessError(code, ["rclone", "lsjson"], stderr="403 forbidden"))
    with pytest.raises(RuntimeError, match="403 forbidden"):
        _CURATED_CALLS[call](SceneSource(_FakeClient()))


@pytest.mark.parametrize("call", sorted(_CURATED_CALLS))
def test_curated_listing_does_not_memoize_a_failure(monkeypatch, call):
    """scene_video's old comment claimed an unreachable bucket "propagates uncached" — it did not,
    because list_directory returned [], which memoized. A fault must re-attempt on the next call."""
    _fake_run(monkeypatch, error=subprocess.CalledProcessError(5, ["rclone", "lsjson"], stderr="temporary"))
    client = _FakeClient()
    src = SceneSource(client)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            _CURATED_CALLS[call](src)
    assert len(client.cmds) == 2
    assert not src._listing_cache


def test_list_scenes_logs_when_the_curated_bucket_is_empty(monkeypatch, caplog):
    """An empty bucket is a genuine answer, so it returns [] — but it must say so, because the
    driver's "No scenes to process" exit is otherwise indistinguishable from a broken listing."""
    _fake_run(monkeypatch, listings={})
    with caplog.at_level(logging.INFO, logger="collab_splats.remote.sources"):
        assert SceneSource(_FakeClient()).list_scenes() == []
    assert caplog.records, "an empty curated bucket must be logged"


def test_list_processed_scenes_reads_processed_bucket(monkeypatch):
    # Same two negative fixtures as list_scenes: a non-scene dir and a scene-named file must both
    # be dropped, so neither the IsDir check nor SCENE_ID_RE can be removed unnoticed.
    _fake_run(
        monkeypatch,
        listings={
            f"collab-data:{PROCESSED_BUCKET}": _dirs("2026_07_21-rats-C0100", "tmp", "2026_07_20-birds-C0043")
            + _files("2026_05_07-birds-clip_03.mp4")
        },
    )
    assert SceneSource(_FakeClient()).list_processed_scenes() == [
        "2026_07_20-birds-C0043",
        "2026_07_21-rats-C0100",
    ]


def test_has_processed_true_when_listing_nonempty(monkeypatch):
    _fake_run(monkeypatch, listings={f"collab-data:{PROCESSED_BUCKET}/s": _files("transforms.json")})
    assert SceneSource(_FakeClient()).has_processed("s") is True


def test_list_localization_dbs_returns_extractor_dirs(monkeypatch):
    _fake_run(
        monkeypatch,
        listings={
            f"collab-data:{PROCESSED_BUCKET}/s/feedforward.zarr/local_features": _dirs("xfeat", "disk")
            + _files("zarr.json")
        },
    )
    assert SceneSource(_FakeClient()).list_localization_dbs("s") == ["disk", "xfeat"]


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_reports_empty_and_logs_when_path_is_absent(monkeypatch, caplog, probe):
    """An unprocessed scene lists as exit 0 with [] — GCS prefixes are not directories, so there is
    nothing to 404 on. This, not a non-zero exit, is the real "nothing processed yet" answer, and
    it is the only path on which the required skip-and-report log can fire."""
    call, absent_answer = _PROBES[probe]
    _fake_run(monkeypatch, listings={})
    with caplog.at_level(logging.INFO, logger="collab_splats.remote.sources"):
        assert call(SceneSource(_FakeClient())) == absent_answer
    # Skip-and-report: a silent skip is what let a whole sweep look processed-free.
    assert caplog.records, f"{probe} must log the absent path"


@pytest.mark.parametrize("probe", sorted(_PROBES))
@pytest.mark.parametrize("code", _TRANSPORT_EXIT_CODES)
def test_probe_raises_on_any_nonzero_rclone_exit(monkeypatch, probe, code):
    """A broken rclone must abort the caller, not report every scene as unprocessed — otherwise an
    unattended driver re-runs the entire bucket, and the 60s memo poisons the whole sweep."""
    call, _ = _PROBES[probe]
    _fake_run(monkeypatch, error=subprocess.CalledProcessError(code, ["rclone", "lsjson"], stderr="403 forbidden"))
    # The message must carry rclone's own output, or an aborting driver reports no cause.
    with pytest.raises(RuntimeError, match="403 forbidden"):
        call(SceneSource(_FakeClient()))


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_raises_on_a_missing_bucket(monkeypatch, probe):
    """Exit 3 against GCS means the BUCKET is gone or misspelled, not that the scene is unprocessed
    — and both bucket names are fixed constants, so it can only ever be a config fault. Reporting
    it as "nothing processed" is what would let the driver re-reconstruct the entire bucket."""
    call, _ = _PROBES[probe]
    _fake_run(
        monkeypatch,
        error=subprocess.CalledProcessError(3, ["rclone", "lsjson"], stderr="directory not found"),
    )
    with pytest.raises(RuntimeError, match="directory not found"):
        call(SceneSource(_FakeClient()))


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_raises_when_the_rclone_binary_is_missing(monkeypatch, probe):
    """No rclone at all is the most basic transport fault and must not read as "unprocessed"."""
    call, _ = _PROBES[probe]
    _fake_run(monkeypatch, error=FileNotFoundError(2, "No such file or directory: 'rclone'"))
    with pytest.raises(OSError):
        call(SceneSource(_FakeClient()))


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_raises_when_the_client_is_unavailable(monkeypatch, probe):
    """RcloneClient failed to construct: unknown, not empty."""
    call, _ = _PROBES[probe]
    _fake_run(monkeypatch)
    src = SceneSource(_FakeClient())
    src._client = None
    with pytest.raises(RuntimeError, match="not available"):
        call(src)


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_does_not_memoize_a_transport_failure(monkeypatch, probe):
    """The failure must not be cached for the TTL: one blip would poison an entire sweep."""
    call, _ = _PROBES[probe]
    _fake_run(monkeypatch, error=subprocess.CalledProcessError(5, ["rclone", "lsjson"], stderr="temporary"))
    client = _FakeClient()
    src = SceneSource(client)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            call(src)
    assert len(client.cmds) == 2, "a failed probe must re-attempt, not serve a cached failure"
    assert not src._listing_cache


@pytest.mark.parametrize("probe", sorted(_PROBES))
def test_probe_memoizes_a_genuinely_absent_path(monkeypatch, probe):
    """Absent is a fact about the bucket, not a failure, so it memoizes like any other answer —
    this memo is the whole reason _cached exists (the dashboard probes every listed scene)."""
    call, absent_answer = _PROBES[probe]
    _fake_run(monkeypatch, listings={})
    client = _FakeClient()
    src = SceneSource(client)
    assert call(src) == absent_answer
    assert call(src) == absent_answer
    assert len(client.cmds) == 1


def test_scene_video_reports_a_missing_video_once_per_ttl(monkeypatch, caplog):
    """The report belongs inside the producer. Logged outside it, a cached call re-fires the line on
    every hit for the whole TTL — and the dashboard probes the same scene repeatedly."""
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}/s": _files("readme.md")})
    src = SceneSource(_FakeClient())
    with caplog.at_level(logging.INFO, logger="collab_splats.remote.sources"):
        for _ in range(3):
            with pytest.raises(FileNotFoundError):
                src.scene_video("s")
    assert len([r for r in caplog.records if "no video" in r.getMessage()]) == 1


########
# Liveness — "did this one scene fail, or has rclone stopped working?"
########


def test_check_available_true_when_the_curated_bucket_lists(monkeypatch):
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}": _dirs("2026_07_20-birds-C0043")})
    assert SceneSource(_FakeClient()).check_available() is True


def test_check_available_true_on_an_empty_bucket(monkeypatch):
    """Liveness, not content: an empty bucket is a reachable bucket. Returning False here would
    abort a batch of explicitly-named scenes just because nothing else was curated yet."""
    _fake_run(monkeypatch, listings={})
    assert SceneSource(_FakeClient()).check_available() is True


@pytest.mark.parametrize("code", _TRANSPORT_EXIT_CODES)
def test_check_available_false_on_any_rclone_failure(monkeypatch, code):
    _fake_run(monkeypatch, error=subprocess.CalledProcessError(code, ["rclone", "lsjson"], stderr="403"))
    assert SceneSource(_FakeClient()).check_available() is False


def test_check_available_false_when_the_rclone_binary_is_missing(monkeypatch):
    """The probe answers a question; it must never raise the very fault it was asked about."""
    _fake_run(monkeypatch, error=FileNotFoundError(2, "No such file or directory: 'rclone'"))
    assert SceneSource(_FakeClient()).check_available() is False


def test_check_available_false_when_the_client_cannot_be_constructed(monkeypatch):
    """Still no client after the re-attempt: the honest answer is False, not a raise."""
    _fake_run(monkeypatch)

    def _boom():
        raise RuntimeError("rclone is not available")

    monkeypatch.setattr(sources, "RcloneClient", _boom)
    src = SceneSource(_FakeClient())
    src._client = None
    assert src.check_available() is False


def test_check_available_reattempts_client_construction(monkeypatch):
    """A probe must actually probe. Born during a transient rclone fault, _client stays None forever
    and every later call answers from construction time — so a long-lived dashboard session could
    never recover without a restart, however healthy rclone became."""
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}": _dirs("2026_07_20-birds-C0043")})
    src = SceneSource(_FakeClient())
    src._client = None
    recovered = _FakeClient()
    monkeypatch.setattr(sources, "RcloneClient", lambda: recovered)
    assert src.check_available() is True
    # Cached back, so the transfer that follows a recovered probe does not re-discover the client
    assert src._client is recovered


def test_check_available_bypasses_the_listing_memo(monkeypatch):
    """The whole point is to answer about *now*. Routed through _cached, a success recorded before
    the credentials expired would be served for the rest of the TTL and the batch would grind on."""
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}": _dirs("2026_07_20-birds-C0043")})
    client = _FakeClient()
    src = SceneSource(client)
    # A prior successful listing must not be reusable as evidence of liveness.
    src.list_scenes()
    assert src.check_available() is True
    baseline = len(client.cmds)
    # Now the remote dies. The memo still holds the earlier success.
    _fake_run(monkeypatch, error=subprocess.CalledProcessError(5, ["rclone", "lsjson"], stderr="temporary"))
    assert ("list_scenes",) in src._listing_cache, "precondition: the stale success is still memoized"
    assert src.check_available() is False
    assert len(client.cmds) == baseline + 1, "the probe must re-run rclone, not read the memo"


def test_check_available_does_not_populate_the_memo(monkeypatch):
    """It must not seed the cache either, or it would mask a later real listing failure."""
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}": _dirs("2026_07_20-birds-C0043")})
    src = SceneSource(_FakeClient())
    src.check_available()
    assert src._listing_cache == {}


########
# Transfers — one scene id, no reconstruction/ prefix
########


def test_fetch_video_copies_to_dest(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}/s": _files("C0043.MP4")})
    client = _FakeClient()
    local = SceneSource(client).fetch_video("s", dest_dir=tmp_path)
    assert local == tmp_path / "C0043.MP4"
    # cmds[0] is the scene_video lsjson probe; the copyto is cmds[1].
    assert client.cmds[1][:3] == ["copyto", f"collab-data:{CURATED_BUCKET}/s/C0043.MP4", str(local)]


def test_push_outputs_targets_scene_dir_with_excludes(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient()
    SceneSource(client).push_outputs(tmp_path, "s")
    cmd = client.cmds[0]
    assert cmd[0] == "copy"
    assert cmd[2] == f"collab-data:{PROCESSED_BUCKET}/s"
    for pattern in PUSH_EXCLUDES:
        assert "--exclude" in cmd and pattern in cmd


def test_push_excludes_raw_feature_maps():
    """The scene-root 2D cache is regenerable from frames + extractor, and is the bulk of the tree."""
    assert "/semantics/**" in PUSH_EXCLUDES
    # rclone reads a leading slash as "relative to the transfer root". fnmatch has no such notion,
    # so model it: strip the anchor and match the root-relative path.
    assert any(fnmatch.fnmatchcase("semantics/dinov2.zarr/c/0", p.lstrip("/")) for p in PUSH_EXCLUDES)


@pytest.mark.parametrize("name", ("frames.zarr/zarr.json", "frames.zarr/images/c/0/0/0", "frames.zarr"))
def test_push_no_longer_excludes_the_keyframe_store(name):
    """frames.zarr is the sole persistent frame source, so a pulled processed scene must carry it —
    PULL_EXCLUDES already assumed that, and never pushing it made pulling it impossible."""
    assert not any(fnmatch.fnmatchcase(name, p) for p in PUSH_EXCLUDES), name
    assert "frames.zarr/**" not in PUSH_EXCLUDES


@pytest.mark.parametrize("name", _VIDEO_NAME_CASES)
def test_push_excludes_the_source_video_in_any_case(name):
    """The remote driver fetches the video into the dir it pushes, so it must be excluded."""
    # fnmatchcase stands in for rclone's case-sensitive glob; uppercase extensions are the
    # camera-native form (C0043.MP4) and must be excluded just as surely as lowercase.
    assert any(fnmatch.fnmatchcase(name, p) for p in PUSH_EXCLUDES), name


@pytest.mark.parametrize("name", ("sparse_pc.ply", "run_config.yaml", "mesh.obj"))
def test_push_keeps_non_video_outputs(name):
    """The video patterns must not widen into the artifacts the push exists to deliver."""
    assert not any(fnmatch.fnmatchcase(name, p) for p in PUSH_EXCLUDES), name


def test_pull_processed_copies_and_passes_exclude_flags(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient()
    SceneSource(client).pull_processed("s", tmp_path, excludes=("frames.zarr/**",))
    cmd = client.cmds[0]
    # Must be `copy`, never `sync`: sync deletes destination files missing from the source, which
    # would wipe exactly the artifacts a caller asked to skip (the dense arrays in PULL_EXCLUDES).
    assert cmd[:3] == ["copy", f"collab-data:{PROCESSED_BUCKET}/s", str(tmp_path)]
    assert "--exclude" in cmd and "frames.zarr/**" in cmd


def test_pull_zarr_members_includes_each_bare_member(monkeypatch, tmp_path):
    """Members are zarr-root-relative bare names (dashboard _LIFT_MEMBERS), pulled as `<name>/**`."""
    _fake_popen(monkeypatch)
    client = _FakeClient()
    SceneSource(client).pull_zarr_members("s", tmp_path, members=("pixel_indices", "depth"))
    cmd = client.cmds[0]
    # The remote is already rooted at feedforward.zarr, so an --include may not repeat that prefix,
    # and each pattern needs the /** suffix — either mistake matches nothing and "succeeds" empty.
    assert cmd[:3] == [
        "copy",
        f"collab-data:{PROCESSED_BUCKET}/s/feedforward.zarr",
        str(tmp_path / "feedforward.zarr"),
    ]
    includes = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--include"]
    assert includes == ["pixel_indices/**", "depth/**"]


def test_pull_excludes_cover_the_dense_per_pixel_arrays():
    assert PULL_EXCLUDES == (
        "feedforward.zarr/depth/**",
        "feedforward.zarr/world_points/**",
        "feedforward.zarr/confidence/**",
        "feedforward.zarr/conf/**",
        "feedforward.zarr/features/**",
        "feedforward.zarr/pixel_indices/**",
        "feedforward.zarr/images/**",
    )


def test_on_line_receives_streamed_rclone_output(monkeypatch, tmp_path):
    """The progress callback is the whole reuse story for a headless driver — it must be called."""
    _fake_popen(monkeypatch, lines=["Transferred: 1.2M / 3.4M, 42%, 1M/s\n", "\n", "Transferred: done\n"])
    seen = []
    SceneSource(_FakeClient()).pull_processed("s", tmp_path, on_line=seen.append)
    # Lines arrive stripped; blank lines are dropped.
    assert seen == ["Transferred: 1.2M / 3.4M, 42%, 1M/s", "Transferred: done"]


########
# Verify
########


def test_verify_push_true_on_clean_check(monkeypatch, tmp_path):
    _fake_popen(monkeypatch, lines=["0 differences found\n"], returncode=0)
    client = _FakeClient()
    assert SceneSource(client).verify_push(tmp_path, "s") is True
    cmd = client.cmds[0]
    # Operand ORDER is load-bearing, do not "tidy" it: `rclone check SRC DST --one-way` asserts every
    # SRC file exists in DST, so local-first asks "did everything I have get uploaded?". Swapped, a
    # scene that uploaded 3 of 400 files would verify clean — and callers then delete it locally.
    # The remote path is pinned to the same target push_outputs writes; if they drift, verify passes
    # vacuously against a location nothing was ever pushed to.
    assert cmd[:4] == ["check", str(tmp_path), f"collab-data:{PROCESSED_BUCKET}/s", "--one-way"]


def test_verify_push_false_on_nonzero_exit(monkeypatch, tmp_path):
    _fake_popen(monkeypatch, lines=["1 differences found\n"], returncode=1)
    assert SceneSource(_FakeClient()).verify_push(tmp_path, "s") is False


def test_verify_push_streams_progress_and_uses_fast_list(monkeypatch, tmp_path):
    """frames.zarr makes this a per-chunk MD5 comparison over thousands of files, and it gates an
    rmtree — so it must list recursively and must not go silent for the whole window.

    --stats alone is not enough: rclone logs stats at INFO while its default --log-level is NOTICE,
    so the interval lines are filtered out before they reach stdout (verified, rclone v1.53.3-DEV).
    Without --stats-log-level the operator sees nothing at all during the riskiest window.
    """
    _fake_popen(monkeypatch, lines=["0 differences found\n"])
    client = _FakeClient()
    assert SceneSource(client).verify_push(tmp_path, "s") is True
    cmd = client.cmds[0]
    assert "--fast-list" in cmd
    assert "--stats-one-line" in cmd
    assert cmd[cmd.index("--stats-log-level") + 1] == "NOTICE"
    # A stats interval of 0 disables stats entirely, which would put the silence straight back.
    assert cmd[cmd.index("--stats") + 1] not in ("0", "0s")
    # The content comparison must stay a content comparison: it guards a destructive local delete.
    for weakening in ("--size-only", "--ignore-checksum", "--checksum"):
        assert weakening not in cmd


def test_verify_push_announces_the_start_as_a_liveness_line(monkeypatch, tmp_path, caplog):
    """rclone's check stats carry no byte counters, so our own start line is what tells the operator
    which scene is being verified before the interval heartbeats begin."""
    _fake_popen(monkeypatch, lines=["0 differences found\n"])
    with caplog.at_level(logging.INFO, logger="collab_splats.remote.sources"):
        SceneSource(_FakeClient()).verify_push(tmp_path, "s")
    assert any("s" in r.getMessage() for r in caplog.records), "verify must announce that it started"


def test_verify_push_reports_a_content_mismatch_as_a_mismatch(monkeypatch, tmp_path, caplog):
    """rclone's own summary line is the discriminator: a real difference always says "differences
    found". The exit code cannot be used — rclone returns 1 for a mismatch AND for an unreachable
    remote alike (verified, v1.53.3-DEV)."""
    _fake_popen(
        monkeypatch,
        lines=["ERROR : transforms.json: MD5 differ\n", "NOTICE: 1 differences found\n"],
        returncode=1,
    )
    with caplog.at_level(logging.ERROR, logger="collab_splats.remote.sources"):
        assert SceneSource(_FakeClient()).verify_push(tmp_path, "s") is False
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "mismatch" in text.lower()
    assert "could not run" not in text.lower()


def test_verify_push_reports_a_transport_fault_as_unable_to_verify(monkeypatch, tmp_path, caplog):
    """Same exit 1, no comparison ever happened. Logging this as "verify mismatch" sent an operator
    hunting a corrupted upload when the real cause was that rclone could not reach the bucket."""
    _fake_popen(
        monkeypatch,
        lines=['Failed to create file system for "collab-data:": didn\'t find section in config file\n'],
        returncode=1,
    )
    with caplog.at_level(logging.ERROR, logger="collab_splats.remote.sources"):
        assert SceneSource(_FakeClient()).verify_push(tmp_path, "s") is False
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "could not run" in text.lower()
    assert "mismatch" not in text.lower()


def test_verify_push_false_when_the_rclone_binary_is_missing(monkeypatch, tmp_path, caplog):
    """No rclone at all cannot verify anything, so it must block the delete rather than propagate
    past the driver's gate — and it is a "could not run", never a mismatch."""

    def _boom(*a, **k):
        raise FileNotFoundError(2, "No such file or directory: 'rclone'")

    monkeypatch.setattr(sources.subprocess, "Popen", _boom)
    with caplog.at_level(logging.ERROR, logger="collab_splats.remote.sources"):
        assert SceneSource(_FakeClient()).verify_push(tmp_path, "s") is False
    assert "could not run" in " ".join(r.getMessage() for r in caplog.records).lower()


def test_verify_push_false_when_rclone_unavailable(monkeypatch):
    """Cannot-verify must never read as verified — callers gate a destructive local delete on True."""

    # No client: RcloneClient() fails at construction, so _remote raises RuntimeError on resolve.
    def _boom():
        raise RuntimeError("rclone is not available")

    monkeypatch.setattr(sources, "RcloneClient", _boom)
    src = SceneSource()
    assert src._client is None
    assert src.verify_push(Path("/nonexistent/local"), "s") is False


def test_verify_push_uses_the_same_excludes_as_push(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient()
    SceneSource(client).verify_push(tmp_path, "s")
    cmd = client.cmds[0]
    for pattern in PUSH_EXCLUDES:
        assert pattern in cmd


########
# Diagnostic tail — progress must never evict the error that explains the failure
########


def test_streaming_failure_tail_keeps_the_error_over_progress_lines(monkeypatch, tmp_path):
    """A 5-line tail plus `--stats 2s --retries 3` loses the cause: four stats lines and the summary
    are enough to push an early per-object ERROR out, leaving the operator only percentages."""
    seen = []
    lines = [
        "ERROR : sub/chunk.0: Failed to copy: googleapi: Error 403: forbidden",
        *[f"Transferred: 1.0 GiB / 5.6 GiB, {p}%, 45 MiB/s, ETA 1m" for p in (20, 40, 60, 80)],
        "Failed to copy with 1 errors: last error was: 403",
    ]
    _fake_popen(monkeypatch, lines=lines, returncode=1)
    with pytest.raises(RuntimeError, match="googleapi"):
        SceneSource(_FakeClient()).push_outputs(tmp_path, "s", on_line=seen.append)
    # Filtering the tail must not silence the progress callback — that is a different consumer
    assert len([line for line in seen if parse_rclone_percent(line) is not None]) == 4


########
# Push excludes — anchoring, so a pattern for a root dir cannot swallow a nested namesake
########


def test_push_excludes_anchor_the_2d_feature_cache_at_the_scene_root():
    """Unanchored, `semantics/**` would also exclude `<backend>/semantics/**` — the deliverable.

    Measured with rclone v1.53.3-DEV, local->local `copy --dry-run` over a tree holding both
    `features/x` and `sub/feedforward.zarr/features/x`: `--exclude 'features/**'` skipped BOTH,
    while `--exclude '/features/**'` skipped only the root one. The same rule applies here, and
    the stakes are higher: the intended target is Reconstructor.semantics_cache_dir ==
    output_path/"semantics" (regenerable 2D patch maps, depth 1 under the pushed root), while the
    nested namesake `<backend>/semantics/` holds the lifted per-point features the push exists to
    publish. Dropping the slash would silently ship scenes with no semantics.
    """
    assert "/semantics/**" in PUSH_EXCLUDES
    assert "semantics/**" not in PUSH_EXCLUDES, "unanchored: rclone matches it at any depth"
    # Nothing may exclude the lifted pair under the backend dir, at any depth
    assert not any("semantics" in p and not p.startswith("/") for p in PUSH_EXCLUDES)


########
# Cache + misc
########


def test_listing_cache_hits_once(monkeypatch):
    _fake_run(monkeypatch, listings={f"collab-data:{CURATED_BUCKET}": _dirs("2026_07_20-birds-C0043")})
    client = _FakeClient()
    src = SceneSource(client)
    src.list_scenes()
    src.list_scenes()
    assert len(client.cmds) == 1


def test_invalidate_drops_one_key(monkeypatch):
    _fake_run(
        monkeypatch,
        listings={
            f"collab-data:{PROCESSED_BUCKET}": _dirs("2026_07_20-birds-C0043"),
            f"collab-data:{CURATED_BUCKET}": _dirs("2026_07_20-birds-C0043"),
        },
    )
    client = _FakeClient()
    src = SceneSource(client)
    src.list_scenes()
    src.list_processed_scenes()
    assert len(client.cmds) == 2
    src.invalidate(("list_scenes",))
    src.list_processed_scenes()  # still cached
    assert len(client.cmds) == 2
    src.list_scenes()  # dropped, refetches
    assert len(client.cmds) == 3


def test_push_outputs_invalidates_processed_listings(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    _fake_run(
        monkeypatch,
        listings={
            f"collab-data:{PROCESSED_BUCKET}/s": _files("transforms.json"),
            f"collab-data:{PROCESSED_BUCKET}": _dirs("2026_07_20-birds-C0043"),
            f"collab-data:{PROCESSED_BUCKET}/s/feedforward.zarr/local_features": _dirs("disk"),
        },
    )
    client = _FakeClient()
    src = SceneSource(client)
    src.has_processed("s")
    src.list_localization_dbs("s")
    src.list_processed_scenes()
    # Guard against a vacuous pass: each producer key must be in the cache before the push, so a
    # renamed producer key fails here and a dropped invalidate() call fails below.
    keys = [("has_processed", "s"), ("list_localization_dbs", "s"), ("list_processed_scenes",)]
    for key in keys:
        assert key in src._listing_cache
    src.push_outputs(tmp_path, "s")
    for key in keys:
        assert key not in src._listing_cache


def test_fake_client_cmd_matches_the_real_argv_order():
    """The fake must build argv exactly as RcloneClient._cmd does, or every test here is fiction.

    The real _cmd appends extra_args AFTER the operands, so a fake that puts the remote path last
    diverges the moment a remote carries extra_args — and anything keying on cmd[-1] then reads a
    flag as a path. Comparing against the real function keeps the two from drifting apart.
    """
    extra = ["--gcs-bucket-policy-only", "--config", "/etc/rclone.conf"]
    # Called unbound against a stub: _cmd touches only self.extra_args, and constructing a real
    # RcloneClient would shell out to verify the binary and the remote.
    real = RcloneClient._cmd(SimpleNamespace(extra_args=extra), "lsjson", "collab-data:bucket/path")
    fake = _FakeClient(extra_args=extra)._cmd("lsjson", "collab-data:bucket/path")
    assert fake == real
    # Pin the property the fakes actually depend on, so the reason for the assert above survives.
    assert real.index("collab-data:bucket/path") < real.index("--gcs-bucket-policy-only")
    assert real[-1] != "collab-data:bucket/path"


def test_fake_client_cmd_signature_matches_the_real_one():
    """A re-shaped real _cmd must fail here rather than silently bypass the fake.

    Parameter names and kinds only — the real one carries type annotations the fake does not need.
    """

    def _shape(fn):
        return [(p.name, p.kind) for p in inspect.signature(fn).parameters.values()]

    assert _shape(_FakeClient._cmd) == _shape(RcloneClient._cmd)


def test_parse_rclone_percent_extracts_progress():
    assert parse_rclone_percent("Transferred: 1.2M / 3.4M, 42%, 1M/s") == 42
    assert parse_rclone_percent("no percent here") is None


def test_field_session_methods_are_gone():
    """The multi-camera fieldwork layout does not exist under environments-curated."""
    for gone in ("list_field_sessions", "list_rgb_cameras", "list_camera_videos", "fetch_field_video"):
        assert not hasattr(SceneSource, gone)
