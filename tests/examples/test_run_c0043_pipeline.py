import pytest
from pathlib import Path


def _import_archive():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_c0043_pipeline",
        Path(__file__).parent.parent.parent / "examples/run_c0043_pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_archive_output_dir_moves_dir(tmp_path):
    src = tmp_path / "C0043"
    src.mkdir()
    (src / "preproc").mkdir()
    (src / "preproc" / "transforms.json").write_text("{}")

    mod = _import_archive()
    archive_path = mod.archive_output_dir(src)

    assert not src.exists(), "src should be gone after archive"
    assert archive_path.exists(), "archive path should exist"
    assert archive_path.name.startswith("C0043_archived_")
    assert (archive_path / "preproc" / "transforms.json").exists()


def test_archive_output_dir_raises_if_missing(tmp_path):
    mod = _import_archive()
    with pytest.raises(FileNotFoundError):
        mod.archive_output_dir(tmp_path / "nonexistent")


def test_archive_output_dir_raises_on_symlink(tmp_path):
    target = tmp_path / "real_dir"
    target.mkdir()
    link = tmp_path / "C0043"
    link.symlink_to(target)

    mod = _import_archive()
    with pytest.raises(ValueError, match="symlink"):
        mod.archive_output_dir(link)


def test_run_stage_calls_fn_with_kwargs(tmp_path):
    mod = _import_archive()
    calls = []

    def fake_preprocess(overwrite, kwargs):
        calls.append({"overwrite": overwrite, "kwargs": kwargs})

    # Mirrors real usage: run_stage("Preprocess", splatter.preprocess, overwrite=True, kwargs={...})
    mod.run_stage("Test stage", fake_preprocess, overwrite=True, kwargs={"sfm_tool": "hloc"})
    assert len(calls) == 1
    assert calls[0] == {"overwrite": True, "kwargs": {"sfm_tool": "hloc"}}


def test_run_stage_exits_on_exception():
    mod = _import_archive()

    def boom(**kwargs):
        raise RuntimeError("stage failed")

    with pytest.raises(SystemExit) as exc_info:
        mod.run_stage("Failing stage", boom)
    assert exc_info.value.code == 1
