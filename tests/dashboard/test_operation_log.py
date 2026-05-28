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


def test_panel_returns_component():
    import panel as pn
    log = OperationLog()
    result = log.panel()
    assert result is not None
