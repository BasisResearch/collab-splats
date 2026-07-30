"""Shared dashboard threading plumbing: off-IOLoop calls and the serialized video fetch."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module-level (not per-instance): the splats page and the localize page draw from the SAME
# flat curated namespace and cache into the SAME base_dir/<scene>/ path, so a prefetch on one
# page can race a Load-video click on the other for one file. Two racing rclone writers let a
# frame decode from a partial file. The second caller blocks, then hits the exists() fast path.
# Fetches are rare, so one global lock is cheaper than per-path bookkeeping.
_FETCH_LOCK = threading.Lock()


def run_off_loop(
    fetch: Callable[[], Any],
    apply: Callable[[Any], None],
    *,
    label: str,
    doc: Any | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> threading.Thread:
    """Run blocking `fetch()` on a daemon thread; marshal `apply(result)` back to the IOLoop.

    `doc` is captured on the IOLoop by the caller (pn.state.curdoc); None (tests / no server)
    runs `apply` inline. Exceptions in `fetch` are logged; `on_error(exc)` (if given) is
    marshalled back like `apply`, so callers can surface failures in the dashboard log.
    """

    def work() -> None:
        try:
            result = fetch()
        except Exception as exc:
            logger.warning("%s failed", label, exc_info=True)
            if on_error is not None:

                def deliver_err(e: Exception = exc) -> None:
                    on_error(e)

                doc.add_next_tick_callback(deliver_err) if doc is not None else deliver_err()
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


def ensure_local_video(scene: str, *, base_dir: Path, source: Any, op_log: Any, label: str) -> Path:
    """Return the scene video at base_dir/<scene>/<filename>, fetching it once if absent.

    Serialized across every caller by _FETCH_LOCK, which is held over both the exists() probe
    and the fetch so a concurrent caller cannot observe a half-written file. `label` is the
    progress prefix shown in the operations log.
    """
    with _FETCH_LOCK:
        # scene_video is a memoized listing, so naming the file to probe keeps the fast path free.
        local = Path(base_dir) / scene / source.scene_video(scene)
        if local.exists():
            return local
        on_line = op_log.rclone_progress(label)
        return source.fetch_video(scene, local.parent, on_line=on_line)
