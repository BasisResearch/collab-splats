"""
Single serialized worker that runs every heavy dashboard job off the IOLoop.

- Panel/Bokeh serve on one IOLoop thread; a run, load or CUDA query there freezes every page
- one daemon thread serializes jobs: no parallel model loads (GPU/RAM OOM), no shared-state races
- results return to the IOLoop via doc.add_next_tick_callback, where VTK/Panel mutation is safe
- callers capture doc at enqueue time; pn.state.curdoc is thread-local and None on the worker
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

from collab_splats.utils.torch_utils import pytorch_gc

logger = logging.getLogger(__name__)


########
# GpuWorker
########


class GpuWorker:
    """
    One daemon thread draining a job queue.

    - pytorch_gc runs after every job
    - busy stays True while any job is queued, running, or finishing on the IOLoop
    """

    def __init__(self) -> None:
        """
        Start the worker thread on an empty queue.

        - _inflight is an explicit counter, not queue.unfinished_tasks: task_done races _finish
        """
        self._queue: queue.Queue = queue.Queue()
        self.busy = False
        self._inflight = 0
        self._flight_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, name="gpu-worker", daemon=True
        )
        self._thread.start()

    def submit(
        self, job_fn: Callable[[], Any], on_done: Callable[[Any], None], doc: Any
    ) -> None:
        """
        Enqueue a job for the worker thread; its callback runs on the IOLoop.

        - doc None (tests, non-served) runs the job inline and synchronously

        Args:
            job_fn: work to run on the worker thread.
            on_done: receives the job's result or raised exception, on the IOLoop.
            doc: Bokeh document the callback is scheduled on.
        """
        if doc is None:
            result = self._run(job_fn)
            on_done(result)
            return

        # Count before enqueue so any poll between put() and _finish observes busy
        with self._flight_lock:
            self._inflight += 1
            self.busy = True

        self._queue.put((job_fn, on_done, doc))

    @staticmethod
    def _run(job_fn: Callable[[], Any]) -> Any:
        """
        Run a job, returning its result or the raised exception; always gc.
        """
        try:
            return job_fn()
        except Exception as exc:  # surface, never crash the worker
            logger.exception("gpu job failed")
            return exc
        finally:
            pytorch_gc()

    def _loop(self) -> None:
        """
        Worker thread body: run each queued job and marshal its result to the IOLoop.
        """
        while True:
            job_fn, on_done, doc = self._queue.get()

            try:
                result = self._run(job_fn)

                # Marshal the result back to the IOLoop; render and busy reset happen there
                try:
                    doc.add_next_tick_callback(
                        lambda r=result, cb=on_done: self._finish(cb, r)
                    )
                except (RuntimeError, AttributeError):
                    # Destroyed session: drop the result, keep the worker alive for reconnects
                    logger.debug(
                        "dropping result for a destroyed session", exc_info=True
                    )
                    self._job_done()
            finally:
                self._queue.task_done()

    def _finish(self, on_done: Callable[[Any], None], result: Any) -> None:
        """
        IOLoop side: clear busy (if nothing else is in flight), then deliver the result.
        """
        self._job_done()
        on_done(result)

    def _job_done(self) -> None:
        """
        Retire one in-flight job; drop busy only when no job is queued, running, or finishing.
        """
        with self._flight_lock:
            self._inflight -= 1

            if self._inflight == 0:
                self.busy = False
