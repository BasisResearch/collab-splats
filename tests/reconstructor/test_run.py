"""
Stage table dispatch: order, dependencies, skip and refuse rules.
"""

import pytest

from collab_splats.reconstructor import LEAF_STAGES, STAGES, Reconstructor


def _recon(tmp_path, monkeypatch, done=()):
    """Reconstructor whose stage methods record their calls and whose done() reads a set."""
    r = Reconstructor(
        {
            "input_path": str(tmp_path / "v.mp4"),
            "output_path": str(tmp_path / "out"),
            "semantics": {"enabled": False},
        }
    )
    calls = []

    for stage in STAGES:
        monkeypatch.setattr(r, stage, lambda s=stage, **kwargs: calls.append(s))

    monkeypatch.setattr(r, "done", lambda s: s in done)
    return r, calls


def test_stage_order_is_dict_order():
    assert list(STAGES)[:2] == ["preproc", "pointcloud"]
    assert list(STAGES)[-1] == "reconstruction_quality_report"


def test_run_calls_named_stages_in_table_order(tmp_path, monkeypatch):
    r, calls = _recon(tmp_path, monkeypatch)
    r.run(["pointcloud", "preproc"])
    assert calls == ["preproc", "pointcloud"]


def test_unknown_stage_raises(tmp_path, monkeypatch):
    r, _ = _recon(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="unknown stage"):
        r.run(["verify"])


def test_unmet_dependency_raises(tmp_path, monkeypatch):
    r, _ = _recon(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="requires 'pointcloud'"):
        r.run(["mesh"])


def test_dependency_on_disk_is_met(tmp_path, monkeypatch):
    r, calls = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud"})
    r.run(["mesh"])
    assert calls == ["mesh"]


def test_named_leaf_already_done_raises_without_overwrite(tmp_path, monkeypatch):
    r, _ = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud", "mesh"})

    with pytest.raises(ValueError, match="already exists"):
        r.run(["mesh"])


def test_named_leaf_already_done_reruns_with_overwrite(tmp_path, monkeypatch):
    r, calls = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud", "mesh"})
    r.run(["mesh"], overwrite=True)
    assert calls == ["mesh"]


def test_done_non_leaf_is_skipped(tmp_path, monkeypatch):
    r, calls = _recon(tmp_path, monkeypatch, done={"preproc"})
    r.run(["preproc", "pointcloud"])
    assert calls == ["pointcloud"]


def test_default_stages_follow_config(tmp_path, monkeypatch):
    r, calls = _recon(tmp_path, monkeypatch)
    r.config["mesh"]["enabled"] = False
    r.config["localization"]["enabled"] = False
    r.config["pointcloud"]["bundle_adjustment"]["enabled"] = True
    r.run()
    assert calls == ["preproc", "pointcloud", "refine", "reconstruction_quality_report"]


def test_outputs_names_one_marker_per_file_stage(tmp_path):
    r = Reconstructor({"input_path": "x", "output_path": str(tmp_path)})
    assert set(r.outputs) == set(STAGES) - {"localize"}
    assert r.outputs["mesh"] == r.backend_dir / "mesh.ply"


def test_named_done_leaf_refuses_before_any_stage_runs(tmp_path, monkeypatch):
    r, calls = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud", "mesh"})

    with pytest.raises(ValueError, match="already exists"):
        r.run(["semantics", "mesh"])

    assert calls == []


def test_report_is_a_leaf_stage_depending_only_on_pointcloud():
    assert "reconstruction_quality_report" in LEAF_STAGES
    assert STAGES["reconstruction_quality_report"] == ("pointcloud",)
    assert list(STAGES).index("reconstruction_quality_report") > list(STAGES).index(
        "pointcloud"
    )


def test_report_does_not_demote_any_existing_leaf():
    """
    A new dependency edge would silently break another stage's disk re-run.
    """
    for s in ("refine", "semantics", "mesh", "localize"):
        assert s in LEAF_STAGES
