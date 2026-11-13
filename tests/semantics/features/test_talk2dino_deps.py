"""
Talk2DINOv3 remote-code imports: transformers' check_imports refuses the model without them.
"""

import importlib.util

import pytest

from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor


@pytest.mark.parametrize("name", ["clip", "einops", "omegaconf"])
def test_remote_code_import_installed(name):
    assert importlib.util.find_spec(name) is not None, (
        f"{name} missing: Talk2DINOv3 cannot load"
    )


def test_clip_is_openai_clip():
    import clip  # noqa: PLC0415

    # PyPI `clip` is an unrelated package without load/tokenize
    assert callable(clip.load) and callable(clip.tokenize)


@pytest.mark.gpu
def test_talk2dino_constructs_on_cpu():
    extractor = Talk2DinoExtractor(device="cpu")

    assert extractor is not None
