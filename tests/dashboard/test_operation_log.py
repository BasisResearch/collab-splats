import logging

import pytest

from collab_splats.dashboard.operation_log import OperationLog


def test_operation_log_defaults():
    log = OperationLog()
    assert log.current_op == ""
    assert log.progress == 0
    assert log.is_running is False
    assert log.log_lines == []


def test_start_op():
    log = OperationLog()
    log.start_op("Extracting frames")
    assert log.current_op == "Extracting frames"
    assert log.is_running is True
    assert log.progress == 0


def test_update_progress():
    log = OperationLog()
    log.start_op("Test op")
    log.update_progress(50, "halfway")
    assert log.progress == 50
    assert any("halfway" in line for line in log.log_lines)


def test_update_progress_clamps_to_100():
    log = OperationLog()
    log.start_op("Test")
    log.update_progress(150, "overshoot")
    assert log.progress == 100


def test_finish_op():
    log = OperationLog()
    log.start_op("Test")
    log.finish_op()
    assert log.is_running is False
    assert log.progress == 100


def test_error_op():
    log = OperationLog()
    log.start_op("Test")
    log.error_op("something failed")
    assert log.is_running is False
    assert any("something failed" in line for line in log.log_lines)


def test_log_lines_capped_at_100():
    log = OperationLog()
    log.start_op("Test")
    for i in range(150):
        log.update_progress(0, f"line {i}")
    assert len(log.log_lines) <= 100


def test_attach_logging_bridges_module_logs():
    log = OperationLog()
    lg = logging.getLogger("collab_splats.dummy_bridge")
    with log.attach_logging("collab_splats"):
        lg.info("extract_and_cache: 12/50 frames written")
    assert any("12/50 frames" in line for line in log.log_lines)
    # After detach, further records are not captured.
    n = len(log.log_lines)
    lg.info("after detach")
    assert len(log.log_lines) == n


def test_rclone_progress_forwards_percent_to_status():
    """rclone_progress builds an on_line callback that drives the progress bar from --stats lines."""
    log = OperationLog()
    on_line = log.rclone_progress("⬇ pulling from server")
    on_line("Transferred: 1 GiB / 2 GiB, 42%, 10 MiB/s")
    assert log.progress == 42
    assert log.current_op == "⬇ pulling from server"


def test_panel_returns_component():
    import panel as pn

    log = OperationLog()
    result = log.panel()
    assert result is not None


def test_step_logs_start_and_elapsed():
    log = OperationLog()
    with log.step("zarr read"):
        pass
    assert log.log_lines[0] == "zarr read…"
    assert log.log_lines[1].startswith("zarr read done (")
    assert log.log_lines[1].endswith("s)")


def test_step_logs_failure_and_reraises():
    log = OperationLog()
    with pytest.raises(ValueError):
        with log.step("mesh read"):
            raise ValueError("boom")
    assert log.log_lines[-1].startswith("mesh read FAILED (")
    assert "boom" in log.log_lines[-1]


def test_version_bumps_on_mutation_only():
    log = OperationLog()
    v0 = log.version
    log.append_line("a")
    v1 = log.version
    assert v1 > v0
    log.append_line("a")  # consecutive dupe is collapsed -> no bump
    assert log.version == v1
    log.start_op("x")
    log.update_progress(10, "y")
    log.finish_op()
    log.error_op("z")
    assert log.version > v1
