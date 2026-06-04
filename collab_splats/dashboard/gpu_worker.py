"""Single serialized worker that runs all heavy/CUDA dashboard jobs off the IOLoop.

Panel/Bokeh serve on a single asyncio IOLoop thread. Heavy imports (vggt pulls a
module-level torch.compile -> inductor compile-worker pool) and CUDA inference must
never run on that thread: they block the websocket long enough for the bokeh session
token to expire, and the page never renders.

This worker owns ALL such work on one daemon thread, so jobs are serialized (no parallel
model loads -> no GPU/RAM OOM, no shared-state races) and torch.compile warms once. The
result is marshalled back to the IOLoop via doc.add_next_tick_callback, where VTK/panel
mutation is safe. The doc is captured by the caller at enqueue time — never pn.state.curdoc
inside the worker (it is thread-local and None off the IOLoop).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

from collab_splats.utils.torch_utils import pytorch_gc

logger = logging.getLogger(__name__)


class GpuWorker:
    """One daemon thread draining a job queue; pytorch_gc after every job."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self.busy = False
        self._thread = threading.Thread(target=self._loop, name="gpu-worker", daemon=True)
        self._thread.start()

    def submit(self, job_fn: Callable[[], Any], on_done: Callable[[Any], None], doc: Any) -> None:
        """Enqueue job_fn (runs on the worker); on_done(result_or_exception) runs on the IOLoop.

        doc is None (tests / non-served) -> run inline, synchronously, for testability.
        """
        if doc is None:
            result = self._run(job_fn)
            on_done(result)
            return
        self.busy = True
        self._queue.put((job_fn, on_done, doc))

    @staticmethod
    def _run(job_fn: Callable[[], Any]) -> Any:
        """Run a job, returning its result or the raised exception; always gc."""
        try:
            return job_fn()
        except Exception as exc:  # surface, never crash the worker
            logger.exception("gpu job failed")
            return exc
        finally:
            pytorch_gc()

    def _loop(self) -> None:
        while True:
            job_fn, on_done, doc = self._queue.get()
            try:
                result = self._run(job_fn)
                # Marshal the result back to the IOLoop; render + busy reset happen there.
                doc.add_next_tick_callback(lambda r=result, cb=on_done: self._finish(cb, r))
            finally:
                self._queue.task_done()

    def _finish(self, on_done: Callable[[Any], None], result: Any) -> None:
        """Runs on the IOLoop: clear busy then deliver the result to the handler."""
        if self._queue.empty():
            self.busy = False
        on_done(result)

    def wait_idle(self, timeout: float = 5.0) -> None:
        """Test helper: block until the queue is drained (best-effort)."""
        self._queue.join()
