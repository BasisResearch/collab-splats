"""run_localization orchestration with all heavy pieces faked."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
import zarr

import collab_splats.dashboard.pipeline as pipeline
from collab_splats.dashboard.config import LocalizationConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.localization.localizer import LocalizationResult

# Flat curated scene id — the reconstruction being localized against.
SCENE = "2024_02_06-office-vid"

########
# Fakes
########


class _FakeSource:
    def __init__(self):
        self.pulled = False
        self.pushed = False
        self.excludes = None
        self.pull_args = None

    def pull_processed(self, scene, dest, excludes=()):
        self.pulled = True
        self.excludes = excludes
        self.pull_args = (scene, dest)
        # Guard: only materialise under an absolute dest — a swapped-arg call would otherwise
        # create a stray relative directory named after the scene id in the cwd.
        if Path(dest).is_absolute():
            (Path(dest) / "feedforward.zarr").mkdir(parents=True, exist_ok=True)

    def push_outputs(self, out_dir, scene, on_line=None):
        self.pushed = True


class _FakeLocalizer:
    """Mirrors CameraLocalizer's real layout: 2 reconstruction frames + 1 already-localized one.

    _extrinsics is reconstruction-only (length 2) exactly as in the real class, while the public
    extrinsics property joins in the localized pose to align with image_paths / frame_sources /
    ref_frame_indices. Reading the private attr here yields 2 rows against 3 paths — the
    misalignment that must not reach the dashboard.
    """

    def __init__(self, pose):
        self._pose = pose
        self.appended = None
        self._image_paths = [Path("/orig/00000.jpg"), Path("/orig/00001.jpg"), Path("/orig/cam_f000007.jpg")]
        self._extrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        self._localized_extrinsics = [np.full((4, 4), 7.0, dtype=np.float32)]

    @property
    def image_paths(self):
        return list(self._image_paths)

    @property
    def extrinsics(self):
        return np.concatenate([self._extrinsics, np.stack(self._localized_extrinsics)], axis=0)

    @property
    def frame_sources(self):
        return ["reconstruction", "reconstruction", "localized"]

    def localize(self, image, K):
        m = 8
        return LocalizationResult(
            pose=self._pose,
            n_correspondences=m,
            n_inliers=6,
            pts2d=np.zeros((m, 2), np.float32),
            pts3d_matched=np.zeros((m, 3), np.float32),
            inlier_mask=np.ones(m, bool),
            pts2d_ref=np.zeros((m, 2), np.float32),
            ref_frame_indices=np.zeros(m, np.int32),
            query_features=object(),
        )

    def add_localized_frame(
        self, image_path, pose, intrinsics, features, zarr_path=None, extractor_name=None, provenance=None
    ):
        self.appended = provenance


########
# Fixtures
########


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Patch every heavy dependency; return the fake localizer for assertions."""
    fake_localizer = _FakeLocalizer(pose=np.eye(4, dtype=np.float32))

    class _FakeResult:
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        intrinsics = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
        image_paths = [Path("/orig/00000.jpg"), Path("/orig/00001.jpg")]

    monkeypatch.setattr(
        pipeline,
        "_load_feedforward_result",
        lambda out_dir, load_world_points=False, load_images=False: _FakeResult(),
    )
    # MagicMock (not a lambda) so tests can assert on call_args — in particular that
    # frames_zarr is threaded through to the real from_feedforward call site.
    build_localizer_mock = MagicMock(return_value=fake_localizer)
    monkeypatch.setattr(pipeline, "_build_localizer", build_localizer_mock)
    fake_localizer.build_localizer_mock = build_localizer_mock
    monkeypatch.setattr(pipeline, "_stamp_db_provenance", lambda zarr_path, extractor, out_dir: None)
    monkeypatch.setattr(pipeline, "extract_frame", lambda video, idx: np.zeros((48, 64, 3), np.uint8))
    monkeypatch.setattr(pipeline, "_resolve_query_intrinsics", lambda cfg: np.eye(3, dtype=np.float32))
    # Push runs inline (no thread) so the flag is set before assertions
    monkeypatch.setattr(
        pipeline,
        "_push_async",
        lambda source, out_dir, scene, op_log: source.push_outputs(out_dir, scene),
    )
    return fake_localizer


def _run(tmp_path, wired, append=True, source=None):
    source = source or _FakeSource()
    out = pipeline.run_localization(
        query_video=tmp_path / "cam.mp4",
        frame_idx=42,
        scene=SCENE,
        config=LocalizationConfig(append_to_db=append),
        op_log=OperationLog(),
        source=source,
        base_dir=tmp_path,
        provenance={"scene": "2024_02_06-office-query", "frame_idx": 42},
    )
    return out, source


########
# Tests
########


def test_returns_result_and_scene_context(tmp_path, wired):
    out, source = _run(tmp_path, wired)
    assert out.result.pose is not None
    assert out.result.n_inliers == 6
    assert len(out.ref_image_paths) == 3
    assert out.ref_extrinsics.shape == (3, 4, 4)
    assert source.pulled  # zarr absent locally → pulled


def test_ref_paths_and_extrinsics_stay_index_aligned(tmp_path, wired):
    """ref_frame_indices indexes both lists, so they must agree in length — and the localized
    frame's pose must be the one carried through, not a reconstruction row."""
    out, _ = _run(tmp_path, wired)
    assert len(out.ref_image_paths) == out.ref_extrinsics.shape[0] == 3
    assert len(out.frame_sources) == 3
    np.testing.assert_allclose(out.ref_extrinsics[2], np.full((4, 4), 7.0, dtype=np.float32))


def test_append_and_push_on_success(tmp_path, wired):
    out, source = _run(tmp_path, wired, append=True)
    assert wired.appended == {"scene": "2024_02_06-office-query", "frame_idx": 42}
    assert source.pushed


def test_no_append_when_disabled(tmp_path, wired):
    out, source = _run(tmp_path, wired, append=False)
    assert wired.appended is None
    assert not source.pushed


def test_no_append_on_failed_pose(tmp_path, wired):
    wired._pose = None
    out, source = _run(tmp_path, wired, append=True)
    assert out.result.pose is None
    assert wired.appended is None
    assert not source.pushed


def test_ref_paths_remapped_to_local_frames_dir(tmp_path, wired):
    """Reconstruction frames map to frames/; localized frames map to localized_frames/."""
    out, _ = _run(tmp_path, wired)
    out_dir = tmp_path / SCENE
    assert out.ref_image_paths[0] == out_dir / "frames" / "00000.jpg"
    assert out.ref_image_paths[2] == out_dir / "localized_frames" / "cam_f000007.jpg"


def test_pull_uses_minimal_excludes(tmp_path, wired):
    _, source = _run(tmp_path, wired)
    assert source.excludes == pipeline.PULL_EXCLUDES


def test_pull_targets_the_scene_id_and_its_local_dir(tmp_path, wired):
    """pull_processed(scene, dest) is positional and untyped, and mirrors
    push_outputs(local_dir, scene) — a swap is silent, so pin the order."""
    _, source = _run(tmp_path, wired)
    assert source.pull_args == (SCENE, tmp_path / SCENE)


def test_build_localizer_receives_scene_frames_zarr(tmp_path, wired):
    """frames_zarr must be threaded into _build_localizer (-> from_feedforward) as the
    scene's own frames.zarr, not dropped or left implicit — this is what lets a cache
    miss read pixels from the store instead of a possibly-stale ff.image_paths."""
    out_dir = tmp_path / SCENE
    # Present locally, as it would be for a scene that ran preprocessing on this machine.
    (out_dir / "frames.zarr").mkdir(parents=True)

    _run(tmp_path, wired)

    wired.build_localizer_mock.assert_called_once()
    kwargs = wired.build_localizer_mock.call_args.kwargs
    assert kwargs["frames_zarr"] == out_dir / "frames.zarr"


def test_build_localizer_gets_no_frames_zarr_when_absent(tmp_path, wired):
    """Pulled scenes have no local frames.zarr (excluded from PULL_EXCLUDES) — must pass
    None, not a dangling path FrameStore.open() would fail to open."""
    _run(tmp_path, wired)

    kwargs = wired.build_localizer_mock.call_args.kwargs
    assert kwargs["frames_zarr"] is None


def test_stamp_db_provenance_writes_attrs(tmp_path):
    """Real (unpatched) _stamp_db_provenance: run_config provenance lands on the group."""
    out_dir = tmp_path / SCENE
    out_dir.mkdir(parents=True)
    (out_dir / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "env_model": "vggtx",
                "frame_indices": [0, 5, 10],
                "video_ref": f"{SCENE}/vid.mp4",
            }
        )
    )
    zp = out_dir / "feedforward.zarr"
    store = zarr.open(str(zp), mode="a")
    store.require_group("local_features/loma-g/reconstruction")

    pipeline._stamp_db_provenance(zp, "loma-g", out_dir)
    g = zarr.open(str(zp), mode="r")["local_features/loma-g"]
    assert g.attrs["backbone"] == "vggtx"
    assert g.attrs["frame_indices"] == [0, 5, 10]
    assert g.attrs["extractor"] == "loma-g"
