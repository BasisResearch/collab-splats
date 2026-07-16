"""Tests for the serialized GPU job worker."""

import threading
from unittest.mock import patch

from collab_splats.dashboard.gpu_worker import GpuWorker


class _FakeDoc:
    """Stand-in for a Bokeh document: runs the scheduled callback immediately."""

    def __init__(self):
        self.scheduled = []

    def add_next_tick_callback(self, cb):
        self.scheduled.append(cb)
        cb()  # emulate the IOLoop running it


def test_inline_when_doc_none_runs_job_and_on_done():
    w = GpuWorker()
    got = {}
    with patch("collab_splats.dashboard.gpu_worker.pytorch_gc") as gc:
        w.submit(job_fn=lambda: 21 * 2, on_done=lambda r: got.__setitem__("r", r), doc=None)
    assert got["r"] == 42
    gc.assert_called_once()  # cache cleared even on the inline path


def test_doc_path_dispatches_result_to_on_done():
    w = GpuWorker()
    doc = _FakeDoc()
    got = {}
    with patch("collab_splats.dashboard.gpu_worker.pytorch_gc"):
        w.submit(job_fn=lambda: "payload", on_done=lambda r: got.__setitem__("r", r), doc=doc)
        w.wait_idle(timeout=5)
    assert got["r"] == "payload"
    assert doc.scheduled  # render was marshalled to the IOLoop, not run on the worker


def test_job_exception_is_passed_to_on_done_and_worker_survives():
    w = GpuWorker()
    doc = _FakeDoc()
    results = []

    def boom():
        raise ValueError("kaboom")

    with patch("collab_splats.dashboard.gpu_worker.pytorch_gc") as gc:
        w.submit(job_fn=boom, on_done=lambda r: results.append(r), doc=doc)
        w.wait_idle(timeout=5)
        w.submit(job_fn=lambda: "ok", on_done=lambda r: results.append(r), doc=doc)
        w.wait_idle(timeout=5)
    assert isinstance(results[0], ValueError)  # error surfaced, not crashed
    assert results[1] == "ok"  # worker survived to run the next job
    assert gc.call_count == 2  # gc after every job (finally)


def test_worker_survives_dead_document():
    """A destroyed session raises in add_next_tick_callback; the worker must not die."""
    w = GpuWorker()

    class _DeadDoc:
        def add_next_tick_callback(self, cb):
            raise AttributeError("'DocumentCallbackManager' object has no attribute '_change_callbacks'")

    done = threading.Event()

    # First job targets a dead doc -> marshal raises inside the worker loop.
    w.submit(lambda: 1, lambda r: None, _DeadDoc())

    # Second job targets a live doc; it must still run -> proves the worker thread lived.
    class _LiveDoc:
        def add_next_tick_callback(self, cb):
            cb()

    w.submit(lambda: 2, lambda r: done.set(), _LiveDoc())
    assert done.wait(timeout=5.0), "worker thread died after a dead-document marshal"
    assert w.busy is False


def test_pytorch_gc_called_after_each_job():
    w = GpuWorker()
    doc = _FakeDoc()
    with patch("collab_splats.dashboard.gpu_worker.pytorch_gc") as gc:
        w.submit(job_fn=lambda: 1, on_done=lambda r: None, doc=doc)
        w.wait_idle(timeout=5)
    gc.assert_called_once()
