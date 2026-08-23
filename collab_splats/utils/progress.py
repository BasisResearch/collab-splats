"""
Progress reporting: one tqdm-or-callback wrapper shared across modules.
"""

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> Iterator[Any]:
    """
    Yield from an iterable, driving a tqdm bar or an on_progress(done, total) callback.

    A caller with a UI passes on_progress and gets no terminal bar; a caller
    without one gets the bar. Exactly one of the two runs, never both.
    """
    # len() when the caller did not say — an unsized iterable reports total 0,
    # which tqdm renders as an open-ended bar rather than a wrong denominator.
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
