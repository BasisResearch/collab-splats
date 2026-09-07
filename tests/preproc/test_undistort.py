"""
Tests for preproc calibration and undistortion: pycolmap camera, COLMAP framing, cv2 remap.
"""

from pathlib import Path

import cv2
import numpy as np
import pycolmap
import pytest

from collab_splats.preproc import frames as fr
from collab_splats.preproc.undistort import (
    _SIFT_NUM_THREADS,
    calibrate_camera,
    undistort_frames,
)
from collab_splats.preproc.video import iter_frames
from collab_splats.wrapper.reconstructor import _camera_provenance, extract_frames


def _distorted_camera(width=1920, height=1080):
    # Barrel-distorted OPENCV camera (k1=-0.25), centered principal point
    return pycolmap.Camera(
        model="OPENCV",
        width=width,
        height=height,
        params=[1190.4, 1190.4, width / 2, height / 2, -0.25, 0.05, 0.0, 0.0],
    )


def _distort_image(image, camera):
    # cv2.undistortPoints maps distorted->normalized-undistorted; projecting those
    # back through K gives, for each DISTORTED pixel, its undistorted location.
    # Remapping the clean image at those locations SAMPLES the clean image where
    # the undistorted content of each distorted pixel lives — i.e. it distorts it.
    h, w = image.shape[:2]
    K = camera.calibration_matrix()
    dist = np.asarray(camera.params[4:], dtype=np.float64)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)[:, None, :]
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return cv2.remap(image, und[..., 0], und[..., 1], cv2.INTER_LINEAR)


def _write_textured_sequence(out_dir, *, n, width, height):
    # n frames of a high-frequency pattern translating a few pixels per frame
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (height + 4 * n, width + 4 * n, 3), dtype=np.uint8)
    for i in range(n):
        crop = canvas[2 * i : 2 * i + height, 2 * i : 2 * i + width]
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), crop)


########################################################################
# Calibration
########################################################################


def test_calibrate_camera_returns_a_pycolmap_camera(tmp_path):
    # Calibration's output type IS pycolmap's, so nothing round-trips through a dataclass
    images = tmp_path / "images"
    images.mkdir()
    _write_textured_sequence(images, n=12, width=320, height=240)

    cam = calibrate_camera(images, max_frames=12)

    assert isinstance(cam, pycolmap.Camera)
    assert cam.model.name == "OPENCV"
    assert (cam.width, cam.height) == (320, 240)


def test_calibrate_camera_stages_no_image_copies(tmp_path, monkeypatch):
    # pycolmap.extract_features(image_names=...) reads the scene's images/ in place;
    # the old code staged JPEG copies into a tempdir, so any write here means it still does.
    # The COLMAP database still gets a tempdir — that is scratch, not a copy of the images.
    images = tmp_path / "images"
    images.mkdir()
    _write_textured_sequence(images, n=12, width=320, height=240)

    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: pytest.fail("staged a temporary image copy"))
    calibrate_camera(images, max_frames=12)


def test_calibrate_camera_raises_when_registration_is_thin(tmp_path):
    # A featureless sequence cannot calibrate, and says so rather than returning nonsense
    images = tmp_path / "images"
    images.mkdir()
    for i in range(12):
        cv2.imwrite(str(images / f"frame_{i:06d}.png"), np.full((240, 320, 3), 128, np.uint8))

    with pytest.raises(RuntimeError, match="registered"):
        calibrate_camera(images, max_frames=12)


def test_calibrate_camera_rejects_too_few_images(tmp_path):
    # Below the two-view-plus-margin floor there is nothing to constrain k1 k2 p1 p2
    images = tmp_path / "images"
    images.mkdir()
    _write_textured_sequence(images, n=4, width=64, height=48)

    with pytest.raises(ValueError, match="at least 8 images"):
        calibrate_camera(images)


def test_calibrate_camera_caps_sift_threads(tmp_path, monkeypatch):
    # The pycolmap wheel here is CPU-only, so an uncapped num_threads (-1) spawns one
    # SIFT thread per host core and OOM-kills the container on 1080p frames.
    captured = {}

    def fake_extract(database, image_dir, **kwargs):
        captured["extract"] = kwargs["extraction_options"].num_threads

    def fake_match(database, **kwargs):
        captured["match"] = kwargs["matching_options"].num_threads

    monkeypatch.setattr(pycolmap, "extract_features", fake_extract)
    monkeypatch.setattr(pycolmap, "match_exhaustive", fake_match)
    monkeypatch.setattr(pycolmap, "incremental_mapping", lambda *a, **kw: {})

    images = tmp_path / "images"
    images.mkdir()
    for i in range(12):
        (images / f"frame_{i:06d}.png").touch()

    with pytest.raises(RuntimeError, match="no model registered"):
        calibrate_camera(images, max_frames=12)

    assert captured == {"extract": _SIFT_NUM_THREADS, "match": _SIFT_NUM_THREADS}
    assert 0 < _SIFT_NUM_THREADS <= 16


@pytest.mark.slow
def test_calibrate_camera_tutorial_smoke(tmp_path):
    # Real-footage smoke: SIFT + exhaustive + mapper on 20 tutorial frames.
    # Slow (~1-3 min CPU); asserts a sane shared-camera OPENCV solve, not
    # specific distortion values.
    video = Path("data/tutorial/tutorial_example-video.mp4")
    if not video.exists():
        pytest.skip("tutorial video not present")

    # 20 frames over a tighter window keeps enough overlap for exhaustive matching
    images = tmp_path / "images"
    images.mkdir()
    for idx, bgr in iter_frames(video, indices=list(range(0, 400, 20))):
        cv2.imwrite(str(images / f"frame_{idx:06d}.png"), bgr)
    assert len(fr.frame_paths(images)) == 20

    camera = calibrate_camera(images, max_frames=20)

    height, width = cv2.imread(str(fr.frame_paths(images)[0])).shape[:2]
    fx, fy, _cx, _cy, k1, k2, p1, p2 = (float(v) for v in camera.params)
    assert camera.model.name == "OPENCV"
    assert (camera.width, camera.height) == (width, height)
    assert 0 < fx < 4 * width and 0 < fy < 4 * width
    assert abs(k1) < 0.5 and abs(k2) < 0.5
    assert np.isfinite([k1, k2, p1, p2]).all()


########################################################################
# Undistortion
########################################################################


def test_undistort_recovers_synthetic_distortion():
    # Checkerboard so residuals are visible: distort with the known camera, undistort
    # with the same camera, compare against the clean original re-framed onto the new
    # canvas. Focal is preserved, so re-framing is a pure translation of the principal
    # point — anything left over is distortion the round trip failed to remove.
    camera = _distorted_camera(width=640, height=480)
    tile = np.kron(np.indices((12, 16)).sum(0) % 2, np.ones((40, 40))) * 255
    clean = np.repeat(tile.astype(np.uint8)[:, :, None], 3, axis=2)

    restored, new_camera = undistort_frames(_distort_image(clean, camera)[None], camera)

    shift = np.float32(
        [
            [1, 0, new_camera.principal_point_x - camera.principal_point_x],
            [0, 1, new_camera.principal_point_y - camera.principal_point_y],
        ]
    )
    reference = cv2.warpAffine(clean, shift, (new_camera.width, new_camera.height))

    # Interior compare (border interpolation is lossy either way). Measured 2.47 here
    # against 15.95 for the same comparison with the undistortion step skipped.
    diff = np.abs(restored[0][30:-30, 30:-30].astype(int) - reference[30:-30, 30:-30].astype(int))
    assert diff.mean() < 5.0


def test_undistort_frames_keeps_the_focal_and_grows_the_canvas():
    # COLMAP's framing: focal is preserved, the canvas expands to hold the corners
    camera = _distorted_camera()

    out, new_camera = undistort_frames(np.zeros((2, 1080, 1920, 3), np.uint8), camera)

    assert new_camera.model.name == "PINHOLE"
    assert new_camera.focal_length_x == camera.focal_length_x
    assert (new_camera.width, new_camera.height) > (camera.width, camera.height)
    assert out.shape == (2, new_camera.height, new_camera.width, 3)


def test_undistort_frames_straightens_a_line():
    # A row of dots bowed by barrel distortion comes back collinear
    camera = _distorted_camera(width=640, height=480)

    # Project a straight world line THROUGH the distortion, so undistorting must straighten it
    image = np.zeros((480, 640, 3), np.uint8)
    for x in np.linspace(-0.35, 0.35, 9):
        u, v = camera.img_from_cam(np.array([[x, -0.2, 1.0]]))[0]
        cv2.circle(image, (int(round(u)), int(round(v))), 3, (255, 255, 255), -1)

    out, _ = undistort_frames(image[None], camera)

    # Centroid of each blob in the output; a straight line has near-zero y spread
    gray = cv2.cvtColor(out[0], cv2.COLOR_RGB2GRAY)
    count, _, _, centroids = cv2.connectedComponentsWithStats((gray > 128).astype(np.uint8))
    ys = sorted(c[1] for c in centroids[1:])

    assert count - 1 >= 7, "lost blobs — the warp is dropping content"
    assert max(ys) - min(ys) < 2.0, f"line still bowed: y spread {max(ys) - min(ys):.2f} px"


def test_undistort_frames_returns_a_pinhole_camera_with_no_distortion():
    # No distortion params left on the returned camera to apply a second time
    _, new_camera = undistort_frames(np.zeros((1, 1080, 1920, 3), np.uint8), _distorted_camera())

    assert list(new_camera.params[4:]) == []


def test_wrong_frame_dims_raise():
    # Frames that disagree with the camera would silently warp through the wrong map
    with pytest.raises(ValueError, match="camera is"):
        undistort_frames(np.zeros((1, 100, 100, 3), np.uint8), _distorted_camera(640, 480))


def test_unstacked_frames_raise():
    # A single (H, W, 3) frame is not a stack; asarray would leave ndim 3
    with pytest.raises(ValueError, match="undistort_frames"):
        undistort_frames(np.zeros((480, 640, 3), np.uint8), _distorted_camera(640, 480))


########################################################################
# extract_frames at the undistort boundary
########################################################################


def test_extract_frames_dir_no_undistort_no_payload(tmp_path):
    # Default path unchanged: no undistort key in provenance, native dims kept
    src = tmp_path / "imgs"
    src.mkdir()
    cv2.imwrite(str(src / "000.jpg"), np.zeros((480, 640, 3), np.uint8))

    images_dir = tmp_path / "scene" / "images"
    extract_frames(
        input_path=src,
        images_dir=images_dir,
        frame_selection="fps",
        fps=None,
        min_frames=None,
        max_frames=None,
    )
    assert "undistort" not in fr.read_manifest(images_dir)["provenance"]
    assert fr.read_frames(images_dir)[0].shape == (480, 640, 3)


def test_extract_frames_dir_undistorts(tmp_path):
    # The only test of the undistort=True branch: real calibration off the written
    # images/, then a rewrite of the same store at the undistorted dims. What lands on
    # disk must match the camera provenance claims it was written with.
    src = tmp_path / "imgs"
    src.mkdir()
    _write_textured_sequence(src, n=12, width=320, height=240)

    images_dir = tmp_path / "scene" / "images"
    extract_frames(
        input_path=src,
        images_dir=images_dir,
        frame_selection="fps",
        fps=None,
        min_frames=None,
        max_frames=None,
        undistort=True,
    )

    payload = fr.read_manifest(images_dir)["provenance"]["undistort"]
    assert payload["camera"]["model"] == "OPENCV"
    assert payload["undistorted_camera"]["model"] == "PINHOLE"
    assert (payload["camera"]["width"], payload["camera"]["height"]) == (320, 240)

    stack = fr.read_frames(images_dir)
    assert stack.shape[1:3] == (payload["undistorted_camera"]["height"], payload["undistorted_camera"]["width"])


def test_provenance_roundtrip_through_the_manifest(tmp_path):
    # write_frames json.dumps the provenance dict verbatim, and Camera.todict() hands
    # back a CameraModelId enum and an ndarray — neither survives that. _camera_provenance
    # is what makes the payload writable AND readable back as the same camera.
    camera = _distorted_camera(width=64, height=48)

    fr.write_frames(
        tmp_path / "images",
        [np.zeros((48, 64, 3), np.uint8)],
        [{"frame_idx": 0}],
        {"undistort": {"camera": _camera_provenance(camera)}},
    )

    stored = fr.read_manifest(tmp_path / "images")["provenance"]["undistort"]["camera"]
    restored = pycolmap.Camera(**stored)

    assert restored.model.name == camera.model.name
    assert (restored.width, restored.height) == (camera.width, camera.height)
    np.testing.assert_allclose(restored.params, camera.params)
