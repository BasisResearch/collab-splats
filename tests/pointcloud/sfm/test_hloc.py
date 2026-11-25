"""
HlocCreator with a fake `hloc` package: pairs, confs, the mapper leg.
"""

import sys
import types
from pathlib import Path

import pycolmap
import pytest

from collab_splats.pointcloud.sfm.hloc import (
    HlocCreator,
    sequential_pairs,
)
from tests.pointcloud._stubs import make_recon, make_scene

NAMES = ["frame_000000.png", "frame_000009.png", "frame_000030.png", "frame_000057.png"]


def test_sequential_pairs_link_each_frame_to_the_next_n():
    assert sequential_pairs(["a", "b", "c", "d"], 2) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ]


@pytest.fixture
def fake_hloc(monkeypatch):
    """
    Install a fake `hloc` package in sys.modules; returns the recorded calls.

    - signatures mirror hloc @ c13273b, keyword names included
    - match/reconstruction keep the pin's Path handling, so a str where a Path belongs fails
    """
    calls = {"extract": [], "retrieval": [], "exhaustive": [], "match": [], "recon": []}
    confs = {
        k: {"output": f"out-{k}"}
        for k in ("netvlad", "superpoint_max", "superpoint+lightglue", "sift")
    }

    def extract(
        conf,
        image_dir,
        export_dir=None,
        as_half=True,
        image_list=None,
        feature_path=None,
        overwrite=False,
    ):
        calls["extract"].append((conf["output"], image_list))
        path = Path(export_dir, conf["output"] + ".h5")
        path.touch()
        return path

    def retrieval(
        descriptors,
        output,
        num_matched,
        query_prefix=None,
        query_list=None,
        db_prefix=None,
        db_list=None,
        db_model=None,
        db_descriptors=None,
    ):
        calls["retrieval"].append((num_matched, query_list, db_list))
        Path(output).write_text("frame_000000.png frame_000057.png")

    def exhaustive(
        output, image_list=None, features=None, ref_list=None, ref_features=None
    ):
        calls["exhaustive"].append(image_list)
        Path(output).write_text("")

    def match(
        conf,
        pairs,
        features,
        export_dir=None,
        matches=None,
        features_ref=None,
        overwrite=False,
    ):
        calls["match"].append((conf["output"], Path(pairs).read_text()))
        assert not isinstance(features, Path), (
            "a Path feature file needs an explicit matches path"
        )
        path = Path(export_dir, f"{features}_{conf['output']}_{pairs.stem}.h5")
        path.touch()
        return path

    def recon_main(
        sfm_dir,
        image_dir,
        pairs,
        features,
        matches,
        camera_mode=pycolmap.CameraMode.AUTO,
        verbose=False,
        skip_geometric_verification=False,
        min_match_score=None,
        image_list=None,
        image_options=None,
        mapper_options=None,
    ):
        assert features.exists() and pairs.exists() and matches.exists()
        calls["recon"].append(
            {
                "camera_mode": camera_mode,
                "image_list": image_list,
                "image_options": image_options,
                "mapper_options": mapper_options,
            }
        )
        (Path(sfm_dir) / "models" / "0").mkdir(parents=True)
        return make_recon(NAMES)

    mods = {
        "hloc": types.ModuleType("hloc"),
        "hloc.extract_features": types.SimpleNamespace(confs=confs, main=extract),
        "hloc.match_features": types.SimpleNamespace(confs=confs, main=match),
        "hloc.pairs_from_retrieval": types.SimpleNamespace(main=retrieval),
        "hloc.pairs_from_exhaustive": types.SimpleNamespace(main=exhaustive),
        "hloc.reconstruction": types.SimpleNamespace(main=recon_main),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
        if "." in name:
            setattr(mods["hloc"], name.split(".")[1], mod)
    return calls


def test_unknown_pairing_is_refused_at_construction():
    with pytest.raises(ValueError, match="pairing must be one of"):
        HlocCreator(pairing="spatial")


def test_map_returns_hlocs_model(tmp_path, fake_hloc):
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    recon = HlocCreator()._map(images_dir, data_dir, NAMES)
    assert recon.num_reg_images() == len(NAMES)


def test_reconstruct_uses_one_simple_radial_camera_and_the_thread_cap(
    tmp_path, fake_hloc
):
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    HlocCreator(num_threads=3)._map(images_dir, data_dir, NAMES)
    kw = fake_hloc["recon"][0]
    assert kw["camera_mode"] == pycolmap.CameraMode.SINGLE
    assert kw["image_options"] == {"camera_model": "SIMPLE_RADIAL"}
    assert kw["mapper_options"] == {"num_threads": 3}
    assert kw["image_list"] == NAMES


def test_sequential_plus_retrieval_matches_both_pair_lists(tmp_path, fake_hloc):
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    HlocCreator(overlap=1, num_retrieved=20)._map(images_dir, data_dir, NAMES)
    pairs = fake_hloc["match"][0][1].split("\n")
    assert pairs == [
        "frame_000000.png frame_000009.png",
        "frame_000009.png frame_000030.png",
        "frame_000030.png frame_000057.png",
        "frame_000000.png frame_000057.png",
    ]
    # topk over 4 images cannot ask for 20 neighbors: clamped to N-1
    assert [num for num, _, _ in fake_hloc["retrieval"]] == [3]


def test_retrieval_is_restricted_to_this_runs_images(tmp_path, fake_hloc):
    # The descriptor h5 persists across runs and may hold names from an older keyframe set
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    HlocCreator()._map(images_dir, data_dir, NAMES)
    assert fake_hloc["retrieval"] == [(3, NAMES, NAMES)]


def test_sequential_pairing_skips_retrieval(tmp_path, fake_hloc):
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    HlocCreator(pairing="sequential")._map(images_dir, data_dir, NAMES)
    assert fake_hloc["retrieval"] == []
    assert [out for out, _ in fake_hloc["extract"]] == ["out-superpoint_max"]


def test_exhaustive_pairing_uses_hloc_generator(tmp_path, fake_hloc):
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    HlocCreator(pairing="exhaustive")._map(images_dir, data_dir, NAMES)
    assert fake_hloc["exhaustive"] == [NAMES]


def test_no_model_raises(tmp_path, fake_hloc, monkeypatch):
    monkeypatch.setattr(
        sys.modules["hloc.reconstruction"], "main", lambda *a, **k: None
    )
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    with pytest.raises(RuntimeError, match="no model"):
        HlocCreator()._map(images_dir, data_dir, NAMES)


def test_missing_hloc_names_the_setup_script(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hloc", None)
    data_dir, images_dir = make_scene(tmp_path, NAMES)
    with pytest.raises(ImportError, match="setup/hloc.sh"):
        HlocCreator()._map(images_dir, data_dir, NAMES)
