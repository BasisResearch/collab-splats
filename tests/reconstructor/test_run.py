"""
Stage table dispatch: order, dependencies, skip and refuse rules.
"""

import pytest

import collab_splats.reconstructor as reconstructor
from collab_splats.reconstructor import (
    GATED_WEIGHTS,
    LEAF_STAGES,
    STAGES,
    Reconstructor,
)


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


def _record_hf_checks(monkeypatch):
    """Replace the gated-weights preflight with a recorder; returns the list of (repo, file) checks."""
    checks = []
    monkeypatch.setattr(
        reconstructor, "check_hf_access", lambda *gated: checks.append(gated)
    )
    return checks


def test_pointcloud_run_checks_gated_weights_before_preproc(tmp_path, monkeypatch):
    r, calls = _recon(tmp_path, monkeypatch)
    checks = []

    def check(*gated):
        checks.append(gated)
        assert calls == []

    monkeypatch.setattr(reconstructor, "check_hf_access", check)
    r.run(["preproc", "pointcloud"])
    assert checks == [GATED_WEIGHTS["vggt_omega"]]


def test_preflight_error_stops_before_any_stage(tmp_path, monkeypatch):
    r, calls = _recon(tmp_path, monkeypatch)

    def deny(*gated):
        raise RuntimeError("No access")

    monkeypatch.setattr(reconstructor, "check_hf_access", deny)

    with pytest.raises(RuntimeError, match="No access"):
        r.run(["preproc", "pointcloud"])

    assert calls == []


def test_no_preflight_when_pointcloud_done_or_not_planned(tmp_path, monkeypatch):
    r, _ = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud"})
    checks = _record_hf_checks(monkeypatch)
    r.run(["preproc", "pointcloud"])
    r.run(["mesh"])
    assert checks == []


def test_preflight_runs_on_overwrite_of_done_pointcloud(tmp_path, monkeypatch):
    r, _ = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud"})
    checks = _record_hf_checks(monkeypatch)
    r.run(["pointcloud"], overwrite=True)
    assert checks == [GATED_WEIGHTS["vggt_omega"]]


def test_no_preflight_with_local_model_path(tmp_path, monkeypatch):
    r, _ = _recon(tmp_path, monkeypatch)
    r.config["pointcloud"]["vggt_omega"] = {"model_path": str(tmp_path / "w.pt")}
    checks = _record_hf_checks(monkeypatch)
    r.run(["preproc", "pointcloud"])
    assert checks == []


def test_no_preflight_for_ungated_backend(tmp_path, monkeypatch):
    r, _ = _recon(tmp_path, monkeypatch)
    r.config["pointcloud"]["backend"] = "mapanything"
    checks = _record_hf_checks(monkeypatch)
    r.run(["preproc", "pointcloud"])
    assert checks == []
