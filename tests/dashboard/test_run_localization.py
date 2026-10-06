"""run_localization orchestration with all heavy pieces faked."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
import zarr
from PIL import Image

import collab_splats.dashboard.pipeline as pipeline
from collab_splats.dashboard.config import LocalizationConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.localization.localizer import LocalizationResult
from collab_splats.utils.io import read_image

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
            (Path(dest) / "pointcloud.zarr").mkdir(parents=True, exist_ok=True)

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
            query_intrinsics=K,
        )

    def add_localized_frame(self, image_path, pose, zarr_path=None, extractor_name=None, provenance=None):
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
        lambda out_dir, load_world_points=False: _FakeResult(),
    )
    monkeypatch.setattr(pipeline, "_build_localizer", MagicMock(return_value=fake_localizer))
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


def test_ref_paths_remapped_to_local_images_dir(tmp_path, wired):
    """Reconstruction frames map to images/; localized frames map to localized_frames/."""
    out, _ = _run(tmp_path, wired)
    out_dir = tmp_path / SCENE
    assert out.ref_image_paths[0] == out_dir / "images" / "00000.jpg"
    assert out.ref_image_paths[2] == out_dir / "localized_frames" / "cam_f000007.jpg"


def test_pull_uses_minimal_excludes(tmp_path, wired):
    _, source = _run(tmp_path, wired)
    assert source.excludes == pipeline.PULL_EXCLUDES


def test_pull_targets_the_scene_id_and_its_local_dir(tmp_path, wired):
    """pull_processed(scene, dest) is positional and untyped, and mirrors
    push_outputs(local_dir, scene) — a swap is silent, so pin the order."""
    _, source = _run(tmp_path, wired)
    assert source.pull_args == (SCENE, tmp_path / SCENE)


def test_build_localizer_refuses_a_scene_without_an_images_store(tmp_path, monkeypatch):
    """A scene with no images/ raises before the matcher loads or a frame is read."""
    matcher = MagicMock()
    monkeypatch.setattr(pipeline, "LocalMatcher", matcher)
    result = MagicMock(image_paths=["frame_000000"])

    with pytest.raises(FileNotFoundError, match="scene has no images/ store"):
        pipeline._build_localizer(
            result,
            LocalizationConfig(matcher="xfeat"),
            tmp_path / "db.zarr",
            OperationLog(),
            images_dir=tmp_path / "images",
        )

    matcher.assert_not_called()


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
    zp = out_dir / "pointcloud.zarr"
    store = zarr.open(str(zp), mode="a")
    store.require_group("local_features/loma-g/reconstruction")

    pipeline._stamp_db_provenance(zp, "loma-g", out_dir)
    g = zarr.open(str(zp), mode="r")["local_features/loma-g"]
    assert g.attrs["backbone"] == "vggtx"
    assert g.attrs["frame_indices"] == [0, 5, 10]
    assert g.attrs["extractor"] == "loma-g"


def test_stem_ids_resolve_to_store_files(tmp_path):
    """Reconstruction ids are the zarr's stems; each resolves to the images/ file of that stem."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images_dir / "frame_000042.png")

    (resolved,) = pipeline._local_ref_paths(tmp_path, ["frame_000042"], ["reconstruction"])

    assert resolved == images_dir / "frame_000042.png"


def test_ids_not_on_disk_keep_their_name(tmp_path):
    """A reconstruction id with no images/ file, or a non-numeric stem tail, keeps its own name."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images_dir / "frame_000042.png")

    resolved = pipeline._local_ref_paths(
        tmp_path,
        ["/remote/images/frame_000007.jpg", "/remote/images/frame_abc.jpg", "/remote/q/query.png"],
        ["reconstruction", "reconstruction", "localized"],
    )

    assert resolved == [
        images_dir / "frame_000007.jpg",
        images_dir / "frame_abc.jpg",
        tmp_path / "localized_frames" / "query.png",
    ]


def test_appended_query_frame_is_lossless_png(tmp_path, wired, monkeypatch):
    """The query frame written for the DB decodes byte-identical to the array localized."""
    noise = np.random.default_rng(0).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(pipeline, "extract_frame", lambda video, idx: noise)

    _run(tmp_path, wired, append=True)

    (saved,) = (tmp_path / SCENE / "localized_frames").iterdir()
    assert saved.suffix == ".png"
    np.testing.assert_array_equal(np.asarray(Image.open(saved)), noise)


def test_build_localizer_decodes_ref_frames_ignoring_exif_orientation(tmp_path, monkeypatch):
    """DB ref frames decode through read_frames in stored-pixel order, as the feedforward loaders do."""
    # 20x40 JPEG store frame tagged EXIF orientation 6: an EXIF-following decode would give 40x20
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    jpg = images_dir / "frame_000000.jpg"
    pixels = np.zeros((20, 40, 3), np.uint8)
    pixels[:, :10] = 255
    im = Image.fromarray(pixels)
    exif = im.getexif()
    exif[0x0112] = 6
    im.save(jpg, exif=exif, quality=95)

    # Capture the images genexpr handed to from_pointcloud; the matcher is never built for real
    seen = {}

    class _Capture:
        @staticmethod
        def from_pointcloud(result, *, zarr_path, images, **kwargs):
            seen["images"] = list(images)
            return MagicMock()

    monkeypatch.setattr(pipeline, "CameraLocalizer", _Capture)
    monkeypatch.setattr(pipeline, "LocalMatcher", MagicMock())

    result = MagicMock(image_paths=["frame_000000"])
    pipeline._build_localizer(
        result, LocalizationConfig(matcher="xfeat"), tmp_path / "db.zarr", OperationLog(), images_dir=images_dir
    )

    (decoded,) = seen["images"]
    assert decoded.shape == (20, 40, 3)
    np.testing.assert_array_equal(decoded, read_image(jpg))


def test_build_localizer_reads_store_frames_lazily_in_zarr_order(tmp_path, monkeypatch):
    """images/ frames pair with zarr rows by frame index, and none is read until the localizer draws."""
    # Four store frames, each a constant equal to its frame index; the zarr keeps three, reordered
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    for idx in (0, 3, 5, 9):
        Image.fromarray(np.full((4, 6, 3), idx, np.uint8)).save(images_dir / f"frame_{idx:06d}.png")

    # Count decodes, and capture the genexpr before and after it is drawn
    reads = []
    real_read_frames = pipeline.fr.read_frames

    def counting_read_frames(dir, idxs=None, **kwargs):
        reads.append(list(idxs))
        return real_read_frames(dir, idxs, **kwargs)

    monkeypatch.setattr(pipeline.fr, "read_frames", counting_read_frames)
    seen = {}

    class _Capture:
        @staticmethod
        def from_pointcloud(result, *, zarr_path, images, **kwargs):
            seen["reads_before"] = len(reads)
            seen["pixels"] = [int(image[0, 0, 0]) for image in images]
            return MagicMock()

    monkeypatch.setattr(pipeline, "CameraLocalizer", _Capture)
    monkeypatch.setattr(pipeline, "LocalMatcher", MagicMock())

    result = MagicMock(image_paths=["frame_000009", "frame_000000", "frame_000005"])
    pipeline._build_localizer(
        result, LocalizationConfig(matcher="xfeat"), tmp_path / "db.zarr", OperationLog(), images_dir=images_dir
    )

    assert seen["reads_before"] == 0
    assert seen["pixels"] == [9, 0, 5]
