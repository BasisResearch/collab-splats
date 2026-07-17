"""Tests for the off-IOLoop helper."""

from collab_splats.dashboard.async_utils import run_off_loop


def test_run_off_loop_applies_result_inline_when_no_doc():
    seen = []
    t = run_off_loop(lambda: 42, seen.append, label="t", doc=None)
    t.join(timeout=5)
    assert seen == [42]


def test_run_off_loop_swallows_fetch_error_and_skips_apply():
    seen = []

    def boom():
        raise RuntimeError("network down")

    t = run_off_loop(boom, seen.append, label="t", doc=None)
    t.join(timeout=5)
    assert seen == []  # apply not called, no exception propagated


def test_on_error_called_with_exception():
    errors = []

    def fetch():
        raise RuntimeError("rclone down")

    t = run_off_loop(fetch, lambda r: None, label="x", doc=None, on_error=errors.append)
    t.join(timeout=5)
    assert len(errors) == 1
    assert "rclone down" in str(errors[0])


def test_error_without_handler_still_swallowed():
    t = run_off_loop(lambda: 1 / 0, lambda r: None, label="x", doc=None)
    t.join(timeout=5)  # must not raise
