"""
Unit tests for the instantsfm / instantsfm_nodepth eval conditions (mapper and VDA faked, no GPU).
"""

import sys
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from PIL import Image

from collab_splats.pointcloud.sfm import base as base_mod
from collab_splats.pointcloud.sfm.instantsfm import InstantSfMCreator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "scripts"))
import eval as eval_gt  # noqa: E402


def _model(names):
    """
    A mapper-shaped model: 30 grid points seen by every image; image i sits at tx = 0.01 * i.

    - dense enough for align_depth's 20-observation floor; tx tells which model image a pose is
    """
    model = pycolmap.Reconstruction()
    camera = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
    model.add_camera_with_trivial_rig(camera)

    # 6 x 5 grid on the z = 5 plane, projected exactly into every view
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-0.8, 0.8, 5))
    grid = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 5.0)], axis=1)
    for i, name in enumerate(names):
        shift = np.array([0.01 * i, 0.0, 0.0])
        xy = 50.0 * (grid + shift)[:, :2] / 5.0 + np.array([32.0, 24.0])
        image = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        image.points2D = [pycolmap.Point2D(pt) for pt in xy]
        model.add_image_with_trivial_frame(image, pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), shift))

    # Every grid point tracked through every image
    for j, xyz in enumerate(grid):
        track = pycolmap.Track()
        for i in range(len(names)):
            track.add_element(i + 1, j)
        model.add_point3D(xyz, track, np.zeros(3, np.uint8))
    return model


def _tx_rows(extr):
    """
    Model image index per pose row, read back from tx.
    """
    return np.round(extr[:, 0, 3] * 100).tolist()


def _write_pngs(image_dir, n, pattern="{i:06d}.png"):
    """
    n RGB PNGs on the model's 64x48 camera grid, pixel value 10 * i.
    """
    image_dir.mkdir()
    paths = []
    for i in range(n):
        p = image_dir / pattern.format(i=i)
        Image.fromarray(np.full((48, 64, 3), i * 10, dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


@pytest.fixture
def fake_sfm(monkeypatch):
    """
    Fake the VDA depth and the InstantSfM mapper; the rest of create_pointcloud() runs for real.

    - set seen["model"] to the image names the fake mapper registers, in registration order
    - _model puts image i at tx = 0.01 * i, so a pose's tx tells which model image it came from
    """
    seen = {"model": None, "map": []}

    # Constant metric depth, no VDA model
    def fake_depth(frames, out_dir, names):
        return np.full((len(names), 12, 16), 5.0, np.float32)

    # Mapper leg: record the call and the creator's depth mode
    def fake_map(self, images_dir, out_dir, names):
        seen["map"].append({"images_dir": images_dir, "out_dir": out_dir, "names": names, "depths": self.use_depths})
        return _model(seen["model"])

    monkeypatch.setattr(base_mod, "estimate_depth", fake_depth)
    monkeypatch.setattr(InstantSfMCreator, "_map", fake_map)
    return seen


def test_validate_condition_accepts_instantsfm():
    for cond in ["instantsfm", "instantsfm_nodepth"]:
        eval_gt._validate_condition(cond)


def test_run_condition_rejects_submap_size_for_instantsfm(tmp_path):
    with pytest.raises(ValueError, match="submap_size"):
        eval_gt._run_condition("instantsfm", tmp_path, tmp_path / "out", submap_size=50)


def test_run_instantsfm_nodepth_stages_symlinks_and_orders_by_name(tmp_path, fake_sfm):
    paths = _write_pngs(tmp_path / "imgs", 3)

    # Registration order differs from name order — the result must follow names
    fake_sfm["model"] = ["000002.png", "000000.png", "000001.png"]
    out = tmp_path / "out"
    extr, n_loops = eval_gt._run_condition("instantsfm_nodepth", tmp_path / "imgs", out)

    # Mapper saw the staged dir, output_dir as scratch, the sorted names, no depth priors
    (call,) = fake_sfm["map"]
    assert call == {
        "images_dir": out / "images",
        "out_dir": out,
        "names": [p.name for p in paths],
        "depths": False,
    }
    staged = sorted((out / "images").iterdir())
    assert [p.name for p in staged] == [p.name for p in paths]
    assert all(p.is_symlink() and p.resolve() == src.resolve() for p, src in zip(staged, paths))

    # (N,4,4) float32 w2c, tx encodes registration idx: sorted by name -> 000000 first (idx 1)
    assert n_loops is None
    assert extr.shape == (3, 4, 4) and extr.dtype == np.float32
    assert _tx_rows(extr) == [1.0, 2.0, 0.0]
    assert np.allclose(extr[:, 3], [0, 0, 0, 1])
    assert (out / "colmap" / "sparse" / "0" / "images.bin").exists()


def test_run_instantsfm_depth_condition_feeds_depth_priors(tmp_path, fake_sfm):
    _write_pngs(tmp_path / "imgs", 2)
    fake_sfm["model"] = ["000000.png", "000001.png"]

    extr = eval_gt._run_instantsfm("instantsfm", tmp_path / "imgs", tmp_path / "out")

    assert fake_sfm["map"][0]["depths"] is True
    assert extr.shape == (2, 4, 4)


def test_run_instantsfm_partial_registration_names_missing(tmp_path, fake_sfm):
    _write_pngs(tmp_path / "imgs", 3)
    fake_sfm["model"] = ["000000.png", "000002.png"]

    with pytest.raises(RuntimeError, match="000001"):
        eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs", tmp_path / "out")


def test_run_instantsfm_multi_dot_name_matches_on_full_stem(tmp_path, fake_sfm):
    # Matching is Path.stem (last suffix only), not a split on the first dot: frame.0.png -> frame.0
    _write_pngs(tmp_path / "imgs", 2, pattern="frame.{i}.png")
    fake_sfm["model"] = ["frame.0.png", "frame.1.png"]

    extr = eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs", tmp_path / "out")

    # Both stems resolved -> no partial-registration raise, poses in sorted-name order
    assert extr.shape == (2, 4, 4)
    assert _tx_rows(extr) == [0.0, 1.0]


def test_run_instantsfm_changed_name_set_drops_sift_db(tmp_path, fake_sfm):
    _write_pngs(tmp_path / "imgs", 2)
    fake_sfm["model"] = ["000000.png", "000001.png"]
    out = tmp_path / "out"
    db = out / "colmap" / "instantsfm.db"
    db.parent.mkdir(parents=True)
    db.touch()

    # Stale staged set (different names) -> DB dropped; re-run with the same set keeps it
    (out / "images").mkdir()
    (out / "images" / "999999.png").touch()
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs", out)
    assert not db.exists()
    db.touch()
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs", out)
    assert db.exists()


def test_run_instantsfm_same_names_different_source_drops_caches(tmp_path, fake_sfm):
    # Eval names are positional (000000.png...), so two sources with the same frame
    # count collide on names — link targets must key the SIFT DB and VDA depth cache
    _write_pngs(tmp_path / "imgs_a", 2)
    _write_pngs(tmp_path / "imgs_b", 2)
    fake_sfm["model"] = ["000000.png", "000001.png"]
    out = tmp_path / "out"

    # First run stages imgs_a; plant a DB and depth cache as if it completed
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs_a", out)
    db = out / "colmap" / "instantsfm.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.touch()
    depth_dir = out / "depth_vda"
    depth_dir.mkdir()

    # Same names, different source -> both caches dropped
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs_b", out)
    assert not db.exists()
    assert not depth_dir.exists()

    # Same source again -> caches kept
    db.parent.mkdir(parents=True, exist_ok=True)
    db.touch()
    eval_gt._run_instantsfm("instantsfm_nodepth", tmp_path / "imgs_b", out)
    assert db.exists()
