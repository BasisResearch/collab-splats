"""
One progress helper for tqdm bars and dashboard callbacks alike.
"""

from collab_splats.utils.progress import progress


def test_progress_yields_every_item():
    assert list(progress(range(5), total=5)) == [0, 1, 2, 3, 4]


def test_progress_forwards_done_and_total_to_callback():
    seen = []

    list(progress(range(3), total=3, on_progress=lambda d, t: seen.append((d, t))))

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_progress_infers_total_from_a_sized_iterable():
    seen = []

    list(progress([7, 8], on_progress=lambda d, t: seen.append((d, t))))

    assert seen == [(1, 2), (2, 2)]


def test_progress_handles_an_unsized_iterable():
    seen = []

    out = list(progress(iter([1, 2]), on_progress=lambda d, t: seen.append((d, t))))

    assert out == [1, 2]
    # No length available, so total stays 0 rather than guessing
    assert seen == [(1, 0), (2, 0)]
