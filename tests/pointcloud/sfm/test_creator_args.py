"""
Every sfm creator refuses a bad argument at construction.
"""

import pytest

from collab_splats.pointcloud.sfm import SFM_CREATORS


@pytest.mark.parametrize("backend", sorted(SFM_CREATORS))
@pytest.mark.parametrize("value", [0, True, 1.5, "8"])
def test_num_threads_must_be_a_positive_int(backend, value):
    with pytest.raises(ValueError, match="num_threads"):
        SFM_CREATORS[backend](num_threads=value)


@pytest.mark.parametrize("backend", sorted(SFM_CREATORS))
@pytest.mark.parametrize("value", [0.0, 1.5, True])
def test_min_registered_frac_must_be_in_unit_interval(backend, value):
    with pytest.raises(ValueError, match="min_registered_frac"):
        SFM_CREATORS[backend](min_registered_frac=value)


@pytest.mark.parametrize("backend", sorted(SFM_CREATORS))
@pytest.mark.parametrize("key", ["overlap", "num_retrieved"])
@pytest.mark.parametrize("value", [0, True, 2.0])
def test_pair_counts_must_be_positive_ints(backend, key, value):
    with pytest.raises(ValueError, match=key):
        SFM_CREATORS[backend](**{key: value})


@pytest.mark.parametrize("backend", sorted(SFM_CREATORS))
def test_pairing_must_be_a_known_mode(backend):
    with pytest.raises(ValueError, match="pairing"):
        SFM_CREATORS[backend](pairing="nonsense")


@pytest.mark.parametrize("value", [-1, 2**32, 1.0, True])
def test_instantsfm_random_seed_domain(value):
    with pytest.raises(ValueError, match="random_seed"):
        SFM_CREATORS["instantsfm"](random_seed=value)


@pytest.mark.parametrize("value", [1, 0])
def test_instantsfm_min_num_view_per_track_floor(value):
    with pytest.raises(ValueError, match="min_num_view_per_track"):
        SFM_CREATORS["instantsfm"](min_num_view_per_track=value)


@pytest.mark.parametrize("backend", sorted(SFM_CREATORS))
@pytest.mark.parametrize(
    "pairing", ["sequential", "retrieval", "sequential+retrieval", "exhaustive"]
)
def test_every_pairing_constructs(backend, pairing):
    assert SFM_CREATORS[backend](pairing=pairing).pairing == pairing


@pytest.mark.parametrize("backend", sorted(SFM_CREATORS))
def test_unknown_argument_is_refused(backend):
    # A typo in the config's sfm block reaches the creator as an unknown keyword
    with pytest.raises(TypeError, match="typo_key"):
        SFM_CREATORS[backend](typo_key=1)


@pytest.mark.parametrize("key", ["use_depths", "retriangulate"])
def test_instantsfm_refuses_keys_that_are_not_config_knobs(key):
    # use_depths was removed (depth priors always on); retriangulate misspells retriangulation
    with pytest.raises(TypeError, match=key):
        SFM_CREATORS["instantsfm"](**{key: False})


def test_instantsfm_pairs_exhaustively_by_default():
    assert SFM_CREATORS["instantsfm"]().pairing == "exhaustive"


@pytest.mark.parametrize("value", [None, 0, 2**32 - 1])
def test_instantsfm_random_seed_accepts_null_and_its_domain(value):
    assert SFM_CREATORS["instantsfm"](random_seed=value).random_seed == value


@pytest.mark.parametrize("value", [None, 2, 6])
def test_instantsfm_min_num_view_per_track_accepts_null_and_two_or_more(value):
    assert (
        SFM_CREATORS["instantsfm"](min_num_view_per_track=value).min_num_view_per_track
        == value
    )
