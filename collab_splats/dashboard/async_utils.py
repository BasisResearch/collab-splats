"""Run a blocking call off the Bokeh IOLoop and marshal the result back onto it."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


def run_off_loop(
    fetch: Callable[[], Any],
    apply: Callable[[Any], None],
    *,
    label: str,
    doc: Any | None = None,
) -> threading.Thread:
    """Run blocking `fetch()` on a daemon thread; marshal `apply(result)` back to the IOLoop.

    `doc` is captured on the IOLoop by the caller (pn.state.curdoc); None (tests / no server)
    runs `apply` inline. Exceptions in `fetch` are logged and swallowed so the thread never
    crashes and stale selections fail quietly.
    """

    def work() -> None:
        try:
            result = fetch()
        except Exception:
            logger.warning("%s failed", label, exc_info=True)
            return

        def deliver() -> None:
            apply(result)

        if doc is not None:
            doc.add_next_tick_callback(deliver)
        else:
            deliver()

    thread = threading.Thread(target=work, name=label, daemon=True)
    thread.start()
    return thread
