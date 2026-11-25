import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
)

from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features import BaseFeatureExtractor, ocr_lens
from collab_splats.semantics.features.ocr_lens import (
    OCRLensExtractor,
    WordVocab,
    _lemma,
    load_decoder,
    score_ocr_heads,
    verbalize,
    word_probabilities,
    word_vocabulary,
)

########################################################################
# Vocabulary
########################################################################


class _Tokenizer:
    """Stub tokenizer over a fixed list of raw token strings."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


TOKENS = ["<s>", "▁tree", "▁Trees", "▁rock", "▁the", "ing", "▁gry", "▁sky", "▁42"]


def test_word_vocabulary_default_merges_by_lemma():
    vocab = word_vocabulary(_Tokenizer(TOKENS))
    assert vocab.words == ["tree", "rock", "sky"]
    assert vocab.token_ids.tolist() == [1, 2, 3, 7]
    assert vocab.word_index.tolist() == [0, 0, 1, 2]


def test_word_vocabulary_custom_words_collect_inflections_and_warn(caplog):
    with caplog.at_level(logging.WARNING):
        vocab = word_vocabulary(_Tokenizer(TOKENS), ["tree", "water", "rock"])
    assert vocab.words == ["tree", "rock"]
    assert vocab.token_ids.tolist() == [1, 2, 3]
    assert vocab.word_index.tolist() == [0, 0, 1]
    assert "water" in caplog.text


def test_word_vocabulary_custom_repeated_word_collapses_to_one_row():
    vocab = word_vocabulary(_Tokenizer(TOKENS), ["tree", "rock", "tree"])
    assert vocab.words == ["tree", "rock"]
    assert vocab.word_index.tolist() == [0, 0, 1]


def test_lemma_picks_most_frequent_base_form():
    assert _lemma("was") == "be"
    assert _lemma("has") == "have"
    assert _lemma("trees") == "tree"
    assert _lemma("the") is None


@pytest.mark.parametrize("words", [["Tree", "tree"], ["leave", "leaves"]])
def test_word_vocabulary_custom_words_sharing_a_lemma_raise(words):
    with pytest.raises(ValueError, match="share lemma"):
        word_vocabulary(_Tokenizer(TOKENS), words)


########################################################################
# Decoding
########################################################################


def _identity_decoder(n_tokens: int) -> nn.Module:
    """Decoder whose logits are the input itself."""
    head = nn.Linear(n_tokens, n_tokens, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(n_tokens))
    return nn.Sequential(nn.Identity(), head)


def test_verbalize_sums_tokens_per_word_and_reports_mass():
    vocab = WordVocab(
        words=["a", "b"],
        token_ids=torch.tensor([1, 2, 3]),
        word_index=torch.tensor([0, 0, 1]),
    )
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.3, 0.1]])
    words, top_p, mass = verbalize(probs.log(), _identity_decoder(5), vocab, k=2)

    assert words == [["a", "b"]]
    np.testing.assert_allclose(top_p, [[0.625, 0.375]], rtol=1e-5)
    np.testing.assert_allclose(mass, [0.8], rtol=1e-5)


def test_verbalize_flattens_maps_row_major():
    vocab = WordVocab(
        words=["a", "b"],
        token_ids=torch.tensor([1, 2, 3]),
        word_index=torch.tensor([0, 0, 1]),
    )
    states = torch.randn(6, 5)
    fmap = states.T.reshape(5, 2, 3)

    flat = verbalize(states, _identity_decoder(5), vocab, k=2)
    mapped = verbalize(fmap, _identity_decoder(5), vocab, k=2)

    assert flat[0] == mapped[0]
    np.testing.assert_allclose(flat[1], mapped[1])


def _vocab() -> WordVocab:
    """Two words over tokens 1-3 of a five-token decoder."""
    return WordVocab(
        words=["a", "b"],
        token_ids=torch.tensor([1, 2, 3]),
        word_index=torch.tensor([0, 0, 1]),
    )


def test_verbalize_empty_input_returns_empty_outputs():
    words, top_p, mass = verbalize(
        torch.zeros(0, 5), _identity_decoder(5), _vocab(), k=10
    )

    assert words == []
    assert top_p.shape == (0, 2)
    assert mass.shape == (0,)


def test_verbalize_empty_vocab_raises():
    vocab = WordVocab(
        words=[],
        token_ids=torch.zeros(0, dtype=torch.long),
        word_index=torch.zeros(0, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="no words"):
        verbalize(torch.zeros(1, 5), _identity_decoder(5), vocab)


def test_verbalize_zero_vocab_mass_is_not_nan():
    logits = torch.tensor([[0.0, -1e4, -1e4, -1e4, 0.0]])
    _, top_p, mass = verbalize(logits, _identity_decoder(5), _vocab(), k=2)

    assert not np.isnan(top_p).any()
    np.testing.assert_allclose(mass, [0.0])


def test_word_probabilities_blocks_hold_every_word_and_verbalize_is_their_top_k():
    logits = torch.randn(7, 5)
    blocks = list(word_probabilities(logits, _identity_decoder(5), _vocab(), chunk=3))
    probs = torch.cat([p for p, _ in blocks]).numpy()
    mass = torch.cat([m for _, m in blocks]).numpy()
    words, top_p, top_mass = verbalize(
        logits, _identity_decoder(5), _vocab(), k=2, chunk=3
    )

    assert [len(p) for p, _ in blocks] == [3, 3, 1]
    np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.sort(probs, axis=1)[:, ::-1], top_p, rtol=1e-6)
    np.testing.assert_allclose(mass, top_mass, rtol=1e-6)


def test_verbalize_decodes_ae_codes_like_their_decoded_states():
    torch.manual_seed(0)
    ae = FeatureAutoencoder(input_dim=5, latent_dim=3)
    codes = torch.randn(7, 3)
    with torch.no_grad():
        states = ae.per_point_decode(codes)

    direct = verbalize(states, _identity_decoder(5), _vocab(), k=2, chunk=3)
    via_ae = verbalize(codes, _identity_decoder(5), _vocab(), k=2, chunk=3, ae=ae)

    assert direct[0] == via_ae[0]
    np.testing.assert_allclose(direct[1], via_ae[1], rtol=1e-5)
    np.testing.assert_allclose(direct[2], via_ae[2], rtol=1e-5)


########################################################################
# Extractor on a tiny LlavaNext (real LLaVA-1.6 processor, random 4-layer model)
########################################################################

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"


@pytest.fixture(scope="module")
def tiny_llava(tmp_path_factory) -> Path:
    """A 4-layer, 32-wide LlavaNext saved beside the real LLaVA-1.6 processor; skips if uncached."""
    # add_prefix_space=None: the cached config forces a slow-tokenizer rebuild needing sentencepiece
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, local_files_only=True, add_prefix_space=None
        )
        real = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError:
        pytest.skip(
            "LLaVA-1.6 processor not in the HF cache (set HF_HOME=/workspace/models)"
        )

    torch.manual_seed(0)
    text = LlamaConfig(
        vocab_size=real.text_config.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    vision = CLIPVisionConfig(
        image_size=336,
        patch_size=14,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    config = LlavaNextConfig(
        text_config=text,
        vision_config=vision,
        image_token_index=real.image_token_index,
        image_grid_pinpoints=real.image_grid_pinpoints,
        vision_feature_layer=-1,
    )
    model = LlavaNextForConditionalGeneration(config)

    # A non-unit final norm, so the load_decoder comparison checks the norm weights
    model.model.language_model.norm.weight.data.uniform_(0.5, 1.5)

    out = tmp_path_factory.mktemp("tiny_llava")
    model.save_pretrained(out, max_shard_size="2MB")
    processor.save_pretrained(out)
    return out


# Top-3 of arange(16) = layer 3, heads 3, 2, 1: all above the extractor's layer 1
TINY_SCORES = torch.arange(16, dtype=torch.float32).reshape(4, 4)


@pytest.fixture(scope="module")
def tiny_ext(tiny_llava) -> OCRLensExtractor:
    return OCRLensExtractor(
        model_id=str(tiny_llava),
        layer=1,
        n_heads=3,
        head_scores=TINY_SCORES,
        dtype="float32",
    )


def _chat_inputs(processor, image: Image.Image, device: torch.device) -> dict:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ""}]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs.to(device)


def test_ocr_lens_is_registered():
    assert BaseFeatureExtractor.get("ocr_lens") is OCRLensExtractor


def test_packaged_head_scores_are_mean_per_head():
    asset = (
        Path(ocr_lens.__file__).parent
        / "assets"
        / "llava16_vicuna7b_ocr_head_scores.pt"
    )
    scores = torch.load(asset, weights_only=True)
    assert scores.shape == (32, 32) and scores.dtype == torch.float32
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    # Top-5 heads pinned to the prototype's scores
    flat = scores.reshape(-1)
    top = torch.topk(flat, 5).indices
    assert sorted(top.tolist()) == [670, 921, 941, 947, 1019]


def test_extractor_honors_device_kwarg(tiny_llava):
    """An explicit device places the model and lens there instead of get_device()."""
    ext = OCRLensExtractor(
        model_id=str(tiny_llava),
        layer=1,
        n_heads=3,
        head_scores=TINY_SCORES,
        dtype="float32",
        device="cpu",
    )

    assert ext.lens.device.type == "cpu"
    assert next(ext.model.parameters()).device.type == "cpu"


def test_lens_sums_top_heads_w_o_w_v_before_truncation(tiny_llava, tiny_ext):
    full = LlavaNextForConditionalGeneration.from_pretrained(tiny_llava)
    attn = full.model.language_model.layers[3].self_attn
    head_dim, n_rep = 8, 2

    expected = torch.zeros(32, 32)
    for head in (3, 2, 1):
        w_o = attn.o_proj.weight[:, head * head_dim : (head + 1) * head_dim]
        w_v = attn.v_proj.weight.view(-1, head_dim, 32)[head // n_rep]
        expected += w_o @ w_v

    torch.testing.assert_close(tiny_ext.lens.cpu(), expected, atol=1e-5, rtol=1e-5)


def test_extractor_truncates_after_layer(tiny_ext):
    assert len(tiny_ext.model.model.language_model.layers) == 2


@pytest.mark.parametrize("layer", [-1, 4])
def test_extractor_rejects_out_of_range_layer(tiny_llava, layer):
    with pytest.raises(ValueError, match="layer"):
        OCRLensExtractor(
            model_id=str(tiny_llava),
            layer=layer,
            n_heads=3,
            head_scores=TINY_SCORES,
            dtype="float32",
        )


def test_extractor_rejects_mis_shaped_head_scores(tiny_llava):
    with pytest.raises(ValueError, match="head_scores"):
        OCRLensExtractor(
            model_id=str(tiny_llava),
            layer=1,
            n_heads=3,
            head_scores=torch.rand(32, 32),
            dtype="float32",
        )


def test_forward_keeps_unpadded_high_res_grid(tiny_llava, tiny_ext):
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8))
    [fmap] = tiny_ext.forward([image])

    # Reference: untruncated model, layer-1 output (hidden_states[2]) at the image tokens
    full = LlavaNextForConditionalGeneration.from_pretrained(tiny_llava)
    full.to(tiny_ext.device).eval()
    processor = tiny_ext.processor
    inputs = _chat_inputs(processor, image, tiny_ext.device)
    with torch.no_grad():
        hidden = full(**inputs, output_hidden_states=True).hidden_states[2][0]
    image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    states = hidden[inputs["input_ids"][0] == image_id]

    # Overview (24x24) first, then one newline closing each high-res row
    D, rows, cols = fmap.shape
    assert D == 32
    assert states.shape[0] == 24 * 24 + rows * (cols + 1)
    grid = states[24 * 24 :].reshape(rows, cols + 1, -1)[:, :-1]
    expected = grid @ tiny_ext.lens.T
    expected = expected / expected.pow(2).mean(-1, keepdim=True).sqrt()
    torch.testing.assert_close(
        fmap, expected.permute(2, 0, 1).cpu(), atol=1e-4, rtol=1e-4
    )


def test_load_decoder_matches_extractor_decoder(tiny_llava, tiny_ext):
    vocab = word_vocabulary(tiny_ext.processor.tokenizer, ["tree", "rock", "sky"])
    states = torch.randn(5, 32)

    got = verbalize(states, load_decoder(str(tiny_llava), dtype="float32"), vocab, k=2)
    want = verbalize(states, tiny_ext.decoder, vocab, k=2)

    assert got[0] == want[0]
    np.testing.assert_allclose(got[1], want[1], rtol=1e-5)
    np.testing.assert_allclose(got[2], want[2], rtol=1e-5)


########################################################################
# Head scoring
########################################################################


def test_score_ocr_heads_shape_and_round_trip(tiny_llava, tmp_path, monkeypatch):
    background = Image.new("RGB", (64, 64), (40, 90, 160))
    monkeypatch.setattr(ocr_lens, "_load_backgrounds", lambda: [background])

    scores = score_ocr_heads(str(tiny_llava), n=4, batch_size=2, dtype="float32")

    assert scores.shape == (4, 4)
    assert torch.isfinite(scores).all() and float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0

    # Same seed, same scores
    again = score_ocr_heads(str(tiny_llava), n=4, batch_size=2, dtype="float32")
    assert torch.equal(scores, again)

    # Round trip as a tensor and as a saved path
    torch.save(scores, tmp_path / "scores.pt")

    for head_scores in (scores, tmp_path / "scores.pt"):
        ext = OCRLensExtractor(
            model_id=str(tiny_llava),
            layer=1,
            n_heads=3,
            head_scores=head_scores,
            dtype="float32",
        )
        assert ext.lens.shape == (32, 32)


def _no_load(*args, **kwargs):
    raise AssertionError("model loaded")


def test_score_ocr_heads_rejects_n_beyond_word_pool(tiny_llava, monkeypatch):
    monkeypatch.setattr(ocr_lens, "_load_model", _no_load)

    with pytest.raises(ValueError, match="n must be"):
        score_ocr_heads(str(tiny_llava), n=10**6, dtype="float32")


def test_score_ocr_heads_rejects_missing_font(tiny_llava, monkeypatch):
    monkeypatch.setattr(ocr_lens, "_load_model", _no_load)

    with pytest.raises(OSError):
        score_ocr_heads(str(tiny_llava), n=4, font="NoSuchFont.ttf", dtype="float32")


def test_score_ocr_heads_frees_the_model_when_a_batch_fails(tiny_llava, monkeypatch):
    background = Image.new("RGB", (64, 64), (40, 90, 160))
    monkeypatch.setattr(ocr_lens, "_load_backgrounds", lambda: [background])
    freed = []
    monkeypatch.setattr(ocr_lens, "pytorch_gc", lambda: freed.append(True))

    def boom(*args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(ocr_lens, "_head_probs", boom)

    with pytest.raises(RuntimeError, match="out of memory"):
        score_ocr_heads(str(tiny_llava), n=4, batch_size=2, dtype="float32")

    assert freed == [True]
