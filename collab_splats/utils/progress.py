"""
Progress reporting: one tqdm-or-callback wrapper shared across modules.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from tqdm.auto import tqdm


def progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> Iterator[Any]:
    """
    Items of an iterable, driving a tqdm bar or an on_progress(done, total) callback.

    - on_progress given: no terminal bar (a UI caller); otherwise the bar
    - exactly one of the two runs, never both

    Args:
        iterable: items to pass through.
        total: item count; len(iterable) when omitted, 0 when unsized.
        desc: tqdm bar label.
        on_progress: called with (done, total) after each item.

    Yields:
        Each item of iterable, in order.
    """
    # Unsized iterable reports total 0: tqdm draws an open-ended bar, not a wrong denominator
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = 0

    # Callback path: no bar at all, so a dashboard worker writes nothing to stdout
    if on_progress is not None:
        for done, item in enumerate(iterable, start=1):
            yield item
            on_progress(done, total)
        return

    # Terminal path: tqdm owns the counting, and closes even if the consumer breaks
    bar = tqdm(total=total or None, desc=desc, unit="frame")

    try:
        for item in iterable:
            yield item
            bar.update(1)
    finally:
        bar.close()
