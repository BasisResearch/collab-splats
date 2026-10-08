"""
Unit tests for docs/source/tutorials/tutorial.py.

- inputs: repo-relative committed paths, and none of the retired shared-cache names
- backend: pyvista backend set at load
- tutorial_scene: one shared scene dir; runs only the named stages not yet on disk
- work_dir: a page's scratch dir under the scene, emptied on each call
"""

import runpy
from pathlib import Path

import pytest
import pyvista as pv

from collab_splats.reconstructor import Reconstructor

REPO = Path(__file__).resolve().parents[2]
TUTORIAL = REPO / "docs" / "source" / "tutorials" / "tutorial.py"


@pytest.fixture
def ns(monkeypatch):
    """
    tutorial.py executed in a fresh namespace, with the pyvista backend call recorded.
    """
    calls = []
    monkeypatch.setattr(pv, "set_jupyter_backend", calls.append)
    namespace = runpy.run_path(str(TUTORIAL))
    namespace["_backend_calls"] = calls
    return namespace


def test_paths_are_repo_relative(ns):
    assert ns["REPO_ROOT"] == REPO
    assert ns["VIDEO_PATH"] == REPO / "data/tutorial/tutorial_example-video.mp4"
    assert ns["QUERY_IMAGE"] == REPO / "data/tutorial/tutorial_example-frame.jpg"
    assert ns["REF_FRAME"] == REPO / "data/tutorial/tutorial_ref-frame.jpg"
    assert ns["SCENE_DIR"] == REPO / "data/tutorial_scene"


def test_exposes_no_shared_cache_names(ns):
    """
    The retired names threaded state between pages; their absence is the isolation property.
    """
    for gone in ("OUTPUT_DIR", "IMAGES_DIR", "RECON", "TUTORIAL_CACHE", "set_notebook_backend"):
        assert gone not in ns, f"{gone} should be gone"


def test_backend_set_on_load(ns, monkeypatch):
    assert ns["_backend_calls"] in (["static"], ["trame"])


def test_backend_static_when_headless(monkeypatch):
    calls = []
    monkeypatch.setattr(pv, "set_jupyter_backend", calls.append)
    monkeypatch.setenv("PYVISTA_OFF_SCREEN", "1")
    runpy.run_path(str(TUTORIAL))
    assert calls == ["static"]


@pytest.fixture
def scene_ns(ns, monkeypatch, tmp_path):
    """
    tutorial.py namespace with SCENE_DIR pointed at tmp_path and Reconstructor.run recorded.
    """
    monkeypatch.setitem(ns["tutorial_scene"].__globals__, "SCENE_DIR", tmp_path)
    runs = []
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages=None, overwrite=False: runs.append(stages))
    ns["_runs"] = runs
    return ns


def test_tutorial_scene_uses_the_shared_dir(scene_ns, tmp_path):
    scene = scene_ns["tutorial_scene"]()
    assert Path(scene.config["output_path"]) == tmp_path
    assert Path(scene.config["input_path"]) == scene_ns["VIDEO_PATH"]
    assert scene.config["pointcloud"]["backend"] == "vggt_omega"
    assert scene.config["mesh"]["source"] == "feedforward"
    assert scene.config["preproc"]["max_frames"] == 96
    assert scene.config["splats"]["losses"]["opacity_reg"]["weight"] == 0


def test_tutorial_scene_runs_only_missing_stages(scene_ns, monkeypatch):
    monkeypatch.setattr(Reconstructor, "done", lambda self, stage: stage == "preproc")
    scene_ns["tutorial_scene"]("preproc", "pointcloud", "mesh")
    assert scene_ns["_runs"] == [["pointcloud", "mesh"]]


def test_tutorial_scene_skips_run_when_all_done(scene_ns, monkeypatch):
    monkeypatch.setattr(Reconstructor, "done", lambda self, stage: True)
    scene_ns["tutorial_scene"]("preproc", "pointcloud")
    assert scene_ns["_runs"] == []


def test_tutorial_scene_extractor_selects_the_store(scene_ns):
    default = scene_ns["tutorial_scene"]()
    ocr = scene_ns["tutorial_scene"](extractor="ocr_lens")
    assert default.outputs["semantics"].name == "maskclip_lifted.zarr"
    assert ocr.outputs["semantics"].name == "ocr_lens_lifted.zarr"
    assert ocr.config["semantics"]["max_epochs"] == 20


def test_tutorial_scene_does_not_mutate_scene_config(scene_ns):
    scene_ns["tutorial_scene"](extractor="ocr_lens")
    assert scene_ns["SCENE_CONFIG"]["semantics"]["extractor"] == "maskclip"


def test_work_dir_is_emptied_on_each_call(scene_ns, tmp_path):
    work_dir = scene_ns["work_dir"]
    first = work_dir("refinement")
    (first / "stale.txt").write_text("x")
    second = work_dir("refinement")
    assert second == tmp_path / "work" / "refinement"
    assert list(second.iterdir()) == []
