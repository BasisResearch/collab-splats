"""
OCR-head verbalization lens: LLaVA-1.6 image-token states decoded to English words.

- paper: Feucht, Krojer, Wang, Abrahamsen, Wallace, Bau (2026), "Using OCR Heads to Verbalize
  Image Semantics", arXiv 2609.18823 — https://ocr.baulab.info/
- code: ported from https://github.com/sfeucht/ocr @ 60868b7b (MIT):
  src/sec2__score_heads.py, src/sec2__ocr.py, src/sec3__object_detection.py, src/sec3__lens.ipynb
- ours, not upstream: LLaVA-1.6 AnyRes grid, WordNet vocabulary, RMS-normalized states
"""

import io
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from nltk.corpus import wordnet as wn
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    PreTrainedTokenizerBase,
)
from transformers.image_processing_utils import select_best_resolution
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.utils import cached_file

from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features.base import BaseFeatureExtractor
from collab_splats.utils.image import open_image
from collab_splats.utils.torch_utils import batch_iterator, get_device, pytorch_gc

logger = logging.getLogger(__name__)

########################################################################
# Classes
########################################################################


@dataclass
class WordVocab:
    """
    Decode vocabulary: word-initial tokens grouped into words.

    - token_ids[j] is a tokenizer id belonging to words[word_index[j]]
    - " Tree", " tree", " trees" share one word
    """

    words: list[str]
    token_ids: torch.Tensor
    word_index: torch.Tensor


@BaseFeatureExtractor.register("ocr_lens")
class OCRLensExtractor(BaseFeatureExtractor):
    """
    LLaVA-1.6 image-token states through the OCR-head verbalization lens, per token-grid cell.

    - maps are rms_normalize(lens @ h); lifted maps are embedding averages, for similarity, not words
    - word probabilities: decode per frame, then lift; the decoder's RMSNorm sharpens averaged states
    - `decoder` + `word_vocabulary` turn states into words via `verbalize`

    Args:
        model_id: LLaVA-1.6 hub id or local directory.
        layer: decoder layer whose output is read, 0-based; the model is truncated after it.
        n_heads: top-scoring OCR heads summed into the lens.
        head_scores: (layers, heads) scores, a path to saved ones, or None for the packaged 7B scores.
        dtype: torch dtype name for the model weights.

    Raises:
        ValueError: `layer`, `n_heads` or `head_scores` out of range for the model.
    """

    # AnyRes has no fixed pixel patch; nominal, recorded in the cache attrs only
    patch_size = 14

    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
        layer: int = 17,
        n_heads: int = 102,
        head_scores: Union[torch.Tensor, str, Path, None] = None,
        dtype: str = "float16",
    ) -> None:
        super().__init__(resize_mode="max_size", image_resolution=672)
        self._device = get_device()

        # Default to the packaged scores; load a path to a tensor
        if head_scores is None:
            head_scores = (
                Path(__file__).parent / "assets" / "llava16_vicuna7b_ocr_head_scores.pt"
            )

        if not isinstance(head_scores, torch.Tensor):
            head_scores = torch.load(head_scores, weights_only=True)

        assert isinstance(head_scores, torch.Tensor)

        # Validate against the config before the (7B) weights load
        config = AutoConfig.from_pretrained(model_id)
        text_config = config.text_config
        n_layers = text_config.num_hidden_layers
        n_query = text_config.num_attention_heads

        if not 0 <= layer < n_layers:
            raise ValueError(f"layer must be in [0, {n_layers}), got {layer}")

        if tuple(head_scores.shape) != (n_layers, n_query):
            raise ValueError(
                f"head_scores must be {(n_layers, n_query)}, got {tuple(head_scores.shape)}"
            )

        if not 1 <= n_heads <= n_layers * n_query:
            raise ValueError(
                f"n_heads must be in [1, {n_layers * n_query}], got {n_heads}"
            )

        self.layer = layer

        # Load the model and processor
        self.model = _load_model(model_id, dtype)
        self.processor = load_processor(model_id)
        lm = self.model.model.language_model

        # Lens from the full model's heads, then drop every layer after `layer`
        lens = _build_lens(self.model, head_scores, n_heads)
        lm.layers = lm.layers[: layer + 1]

        # Move to the device; the decoder shares the model's final norm and lm_head
        self.model.to(self._device).eval()
        lens = lens.to(self._device)
        self.register_buffer("lens", lens)
        self.decoder = nn.Sequential(lm.norm, self.model.lm_head)

        # Chat template with the image and an empty text turn, as upstream; same for every image
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": ""}],
            }
        ]
        self._prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self._image_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )

        # The last kept layer's output lands here on every forward
        self._hidden: Optional[torch.Tensor] = None
        lm.layers[layer].register_forward_hook(self._capture)

    def _capture(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        """
        Forward hook: keep the hooked decoder layer's output hidden states.
        """
        self._hidden = output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def _lens_map(self, image: Image.Image) -> torch.Tensor:
        """
        One image's (D, rows, cols) unit-RMS lens states, float32 on CPU.
        """
        # Prompt built in __init__; one image per forward
        inputs = self.processor(
            text=[self._prompt], images=[image], return_tensors="pt"
        )
        inputs = inputs.to(self._device)

        # Truncated forward; the hook keeps the layer output, logits only for the last token
        self.model(**inputs, use_cache=False, logits_to_keep=1)
        assert self._hidden is not None
        states = self._hidden[0, inputs["input_ids"][0] == self._image_id]
        self._hidden = None

        # Unpadded high-res grid: the last rows * (cols + 1) tokens, one newline closing each row
        rows, cols = _anyres_grid(self.processor, inputs)
        grid = states[-(rows * (cols + 1)) :]
        grid = grid.reshape(rows, cols + 1, -1)
        grid = grid[:, :-1]

        # Lens, then unit RMS per state
        lensed = grid.float() @ self.lens.T
        rms = lensed.pow(2).mean(-1, keepdim=True).sqrt()
        rms = rms.clamp_min(torch.finfo(rms.dtype).tiny)
        lensed = lensed / rms
        lensed = lensed.permute(2, 0, 1)
        return lensed.cpu()

    def forward(self, images: list) -> list[torch.Tensor]:
        """
        Lens states on each image's unpadded AnyRes token grid.

        - one forward per image: AnyRes grids differ in shape
        - the 24x24 overview and row-newline tokens are dropped; the grid spans the full image

        Args:
            images: anything `open_image` accepts, one entry per frame.

        Returns:
            One (D, rows, cols) float32 CPU tensor per image.
        """
        maps = []

        for image in images:
            rgb = open_image(image)
            rgb = rgb.convert("RGB")
            maps.append(self._lens_map(rgb))

        return maps


########################################################################
# Vocabulary
########################################################################


def _lemma(word: str) -> Optional[str]:
    """
    Most frequent WordNet base form over noun / verb / adjective readings.

    - usage count: SemCor tagged-corpus attestations of the form, summed over its senses
    - None when unknown or never attested (usage count 0)
    - was -> be, has -> have, trees -> tree; ties go to the shorter form
    - drops fragments ("gry", "les") and function words ("the")
    """
    forms = {
        form for pos in (wn.NOUN, wn.VERB, wn.ADJ) if (form := wn.morphy(word, pos))
    }

    if not forms:
        return None

    # Usage count per form: WordNet sense-tagged corpus (SemCor) attestations, summed over its senses
    counts = {}

    for form in forms:
        lemmas = wn.lemmas(form)
        counts[form] = sum(lemma.count() for lemma in lemmas)

    best = max(forms, key=lambda form: (counts[form], -len(form), form))

    if counts[best] == 0:
        return None

    return best


def word_vocabulary(
    tokenizer: PreTrainedTokenizerBase, words: Optional[list[str]] = None
) -> WordVocab:
    """
    Decode vocabulary over the tokenizer's word-initial (`▁`) ASCII-letter tokens.

    - default: tokens with a WordNet base form, merged by lemma (~3.5k for Llama-2)
    - custom: each word collects tokens matching its lemma or lowercase text; no-token words dropped

    Args:
        tokenizer: the model tokenizer.
        words: custom label set; None for the default vocabulary.

    Returns:
        The WordVocab, in token order (default) or the given order (custom).

    Raises:
        LookupError: WordNet corpus not installed.
        ValueError: two custom words share a lemma or lowercase text.
    """
    # Fail with the fix, not nltk's generic message
    try:
        wn.ensure_loaded()
    except LookupError as e:
        raise LookupError(
            "WordNet corpus missing: python -m nltk.downloader wordnet"
        ) from e

    # Word-initial ASCII tokens with their lemma and lowercase text
    pattern = re.compile(r"^▁([A-Za-z]{2,})$")
    keyed = []

    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        match = pattern.match(token)

        if match is None:
            continue

        text = match.group(1).lower()
        lemma = _lemma(text)
        keyed.append((token_id, lemma, text))

    # Default: every WordNet lemma is a word
    if words is None:
        pairs = [(token_id, lemma) for token_id, lemma, _ in keyed if lemma is not None]
        labels = list(dict.fromkeys(lemma for _, lemma in pairs))

    else:
        # Custom: exact repeats collapse to one word, first occurrence kept
        words = list(dict.fromkeys(words))

        # A token joins the word whose lemma or text equals its own lemma or text
        targets = {}

        for word in words:
            text = word.lower()
            lemma = _lemma(text)

            # Two words sharing a key would silently steal each other's tokens
            for key in (text, lemma or text):
                other = targets.setdefault(key, word)

                if other != word:
                    raise ValueError(f"{word!r} and {other!r} share lemma {key!r}")

        pairs = []

        for token_id, lemma, text in keyed:
            fallback = targets.get(text)
            word = targets.get(lemma, fallback)

            if word is not None:
                pairs.append((token_id, word))

        # Words no token reached are dropped, once, loudly
        found = {word for _, word in pairs}
        missing = [word for word in words if word not in found]

        if missing:
            logger.warning(
                "word_vocabulary: no single-token form for %s, dropped", missing
            )

        labels = [word for word in words if word in found]

    # Index each token to its word's row
    row_of = {word: row for row, word in enumerate(labels)}
    token_ids = torch.tensor([token_id for token_id, _ in pairs], dtype=torch.long)
    word_index = torch.tensor([row_of[word] for _, word in pairs], dtype=torch.long)
    return WordVocab(words=labels, token_ids=token_ids, word_index=word_index)


########################################################################
# Decoding
########################################################################


@torch.no_grad()
def word_probabilities(
    features: torch.Tensor,
    decoder: nn.Module,
    vocab: WordVocab,
    *,
    chunk: int = 4096,
    ae: Optional[FeatureAutoencoder] = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Per-word probabilities of lens states, one block of rows at a time.

    - decoder softmax over the full token vocabulary, token probs summed per word, renormalized
    - ae: codes are decoded one chunk at a time, so (P, D) states never sit in memory

    Args:
        features: (P, D) lens states, or (P, latent) codes when ae is given.
        decoder: final norm + lm_head, from `OCRLensExtractor.decoder` or `load_decoder`.
        vocab: from `word_vocabulary`.
        chunk: rows per softmax block.
        ae: autoencoder that decodes point codes to lens states, or None when features are states.

    Yields:
        (probs, vocab_mass) per block on the decoder's device: (c, n_words) word probabilities
        summing to 1, and (c,) full-vocab probability landing on the vocabulary.
    """
    # Move the vocabulary index onto the decoder's device
    param = next(decoder.parameters())
    token_ids = vocab.token_ids.to(param.device)
    word_index = vocab.word_index.to(param.device)
    n_words = len(vocab.words)
    tiny = torch.finfo(torch.float32).tiny

    # Blocks of lens states; codes are decoded one block at a time
    if ae is not None:
        blocks = ae.iter_decode(features, chunk)
    else:
        blocks = (block for (block,) in batch_iterator(chunk, features))

    for x in blocks:
        # Decode one block of states to full-vocabulary token probabilities
        x = x.to(device=param.device, dtype=param.dtype)
        logits = decoder(x)
        p = logits.float().softmax(-1)

        # Sum token probabilities into words, then renormalize over the vocabulary
        per_word = torch.zeros(len(p), n_words, device=p.device)
        per_word.index_add_(1, word_index, p[:, token_ids])
        in_vocab = per_word.sum(1, keepdim=True)
        denominator = in_vocab.clamp_min(tiny)
        yield per_word / denominator, in_vocab[:, 0]


@torch.no_grad()
def verbalize(
    features: torch.Tensor,
    decoder: nn.Module,
    vocab: WordVocab,
    *,
    k: int = 10,
    chunk: int = 4096,
    ae: Optional[FeatureAutoencoder] = None,
) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    """
    Top-k vocabulary words per lens state.

    - top-k of `word_probabilities`; a (D, H, W) map is flattened row-major first

    Args:
        features: (P, D) lens states, (P, latent) codes when ae is given, or a (D, H, W) map.
        decoder: final norm + lm_head.
        vocab: from `word_vocabulary`.
        k: words per row, capped at the vocabulary size.
        chunk: rows per softmax block.
        ae: decodes point codes to lens states; None when features are states.

    Returns:
        (words, probs, vocab_mass): P lists of k words, (P, k) probs, (P,) mass on the vocabulary.

    Raises:
        ValueError: empty vocabulary.
    """
    # An empty vocabulary has nothing to rank
    n_words = len(vocab.words)

    if n_words == 0:
        raise ValueError("verbalize: vocabulary has no words")

    k = min(k, n_words)

    # Flatten a (D, H, W) map to row-major (H * W, D) states
    if features.ndim == 3:
        features = features.flatten(1).T

    # No rows: empty outputs with the capped k width
    if features.shape[0] == 0:
        return [], np.zeros((0, k), np.float32), np.zeros(0, np.float32)

    # Top-k words of each block's word probabilities, collected on the CPU
    words, probs, mass = [], [], []
    blocks = word_probabilities(features, decoder, vocab, chunk=chunk, ae=ae)

    for normalized, in_vocab in blocks:
        top_p, top_i = normalized.topk(k, dim=-1)
        words += [[vocab.words[j] for j in row] for row in top_i.tolist()]
        top_p = top_p.cpu()
        probs.append(top_p.numpy())
        in_vocab = in_vocab.cpu()
        mass.append(in_vocab.numpy())

    return words, np.concatenate(probs), np.concatenate(mass)


def load_decoder(
    model_id: str = "llava-hf/llava-v1.6-vicuna-7b-hf", dtype: str = "float16"
) -> nn.Sequential:
    """
    Final norm + lm_head of a LLaVA-1.6 checkpoint, without loading the rest of the model.

    - reads only the two tensors from the safetensors shards (~260 MB at fp16)
    - hub id or local `save_pretrained` dir; old and new checkpoint key layouts

    Args:
        model_id: hub id or local directory.
        dtype: torch dtype name.

    Returns:
        nn.Sequential(norm, lm_head), eval mode, on the default device.

    Raises:
        KeyError: when the checkpoint has no final-norm or lm_head weight.
    """
    config = AutoConfig.from_pretrained(model_id)
    text = config.text_config
    norm = LlamaRMSNorm(text.hidden_size, eps=text.rms_norm_eps)
    lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    # Key -> shard file, from the sharded index or a single model.safetensors
    index_path = cached_file(
        model_id,
        "model.safetensors.index.json",
        _raise_exceptions_for_missing_entries=False,
    )

    if index_path is not None:
        index_text = Path(index_path).read_text()
        weight_map = json.loads(index_text)["weight_map"]

    else:
        single = cached_file(model_id, "model.safetensors")

        with safe_open(single, framework="pt") as f:
            weight_map = {key: "model.safetensors" for key in f.keys()}

    # Find the two keys: `language_model.model.norm` (hub) or `model.language_model.norm` (new saves)
    keys = []

    for pattern in (r"language_model\.(model\.)?norm\.weight$", r"lm_head\.weight$"):
        compiled = re.compile(pattern)
        matches = (key for key in weight_map if compiled.search(key))
        key = next(matches, None)

        if key is None:
            raise KeyError(
                f"load_decoder: no checkpoint key matches {pattern!r} in {model_id}"
            )

        keys.append(key)

    # Copy each tensor out of its shard
    for module, key in zip((norm, lm_head), keys):
        shard = cached_file(model_id, weight_map[key])

        with safe_open(shard, framework="pt") as f:
            weight = f.get_tensor(key)

        with torch.no_grad():
            module.weight.copy_(weight)

    # Cast and place the decoder
    decoder = nn.Sequential(norm, lm_head)
    device = get_device()
    torch_dtype = getattr(torch, dtype)
    decoder.to(device=device, dtype=torch_dtype)
    return decoder.eval()


########################################################################
# Model and lens
########################################################################


def load_processor(model_id: str) -> LlavaNextProcessor:
    """
    LLaVA-1.6 processor with the fast tokenizer.

    - its `.tokenizer` feeds `word_vocabulary`; `AutoTokenizer` needs sentencepiece for this checkpoint

    Args:
        model_id: hub id or local directory.

    Returns:
        The checkpoint's processor.
    """
    # add_prefix_space=None: the hub config forces a slow-tokenizer rebuild needing sentencepiece
    return AutoProcessor.from_pretrained(model_id, add_prefix_space=None)


def _load_model(model_id: str, dtype: str) -> LlavaNextForConditionalGeneration:
    """
    Full LLaVA-1.6 model in `dtype`, still on the CPU.
    """
    torch_dtype = getattr(torch, dtype)
    return LlavaNextForConditionalGeneration.from_pretrained(
        model_id, dtype=torch_dtype
    )


@torch.no_grad()
def _build_lens(
    model: LlavaNextForConditionalGeneration, head_scores: torch.Tensor, n_heads: int
) -> torch.Tensor:
    """
    Sum of W_O W_V over the top-scoring heads: the (D, D) verbalization lens, float32 on CPU.

    - GQA-aware: each kv head serves n_heads / n_kv_heads query heads (Vicuna has no GQA)
    - accumulated in float32 on CPU, before truncation, so heads above `layer` count
    - `head_scores` shape is validated by the caller, before the model loads
    """
    cfg = model.config.text_config
    layers = model.model.language_model.layers
    n_query = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_query
    n_rep = n_query // cfg.num_key_value_heads
    hidden = cfg.hidden_size

    # Top heads over the flattened (layer, head) scores
    lens = torch.zeros((hidden, hidden), dtype=torch.float32)
    flat = head_scores.reshape(-1)
    top = torch.topk(flat, n_heads).indices

    # Accumulate W_O W_V of each top head, slicing its kv head for GQA
    for index in top.tolist():
        layer, head = divmod(index, n_query)
        attn = layers[layer].self_attn
        w_o = attn.o_proj.weight[:, head * head_dim : (head + 1) * head_dim]
        w_v = attn.v_proj.weight.view(-1, head_dim, hidden)[head // n_rep]
        lens += w_o.float() @ w_v.float()

    return lens


def _anyres_grid(
    processor: LlavaNextProcessor, inputs: dict[str, torch.Tensor]
) -> tuple[int, int]:
    """
    (rows, cols) of LLaVA-1.6's unpadded high-res token grid for one processed image.

    - uses the processor's own token-count math (private `_get_unpadded_features`, transformers 4.57)
    """
    orig_h, orig_w = inputs["image_sizes"][0].tolist()
    tile_h, tile_w = inputs["pixel_values"].shape[-2:]
    pinpoints = processor.image_processor.image_grid_pinpoints
    best_h, best_w = select_best_resolution([orig_h, orig_w], pinpoints)

    # High-res token count and row count (one newline per row) for this image
    n_high_res, rows = processor._get_unpadded_features(
        orig_h,
        orig_w,
        tile_h // processor.patch_size,
        tile_w // processor.patch_size,
        best_h // tile_h,
        best_w // tile_w,
    )
    return rows, n_high_res // rows


########################################################################
# Head scoring
########################################################################


def _load_backgrounds() -> list[Image.Image]:
    """
    One mini-imagenet train shard as decoded images (upstream's word-image backgrounds).
    """
    path = hf_hub_download(
        "timm/mini-imagenet", "data/train-00000-of-00013.parquet", repo_type="dataset"
    )
    table = pq.read_table(path, columns=["image"])
    rows = table.column("image").to_pylist()
    images = []

    for row in rows:
        buffer = io.BytesIO(row["bytes"])
        images.append(Image.open(buffer))

    return images


def _contrasting_color(patch: Image.Image, rng: random.Random) -> tuple[int, int, int]:
    """
    Random color at least 90 luminance units from the patch mean; black or white as a fallback.
    """
    # Rec. 601 luma, averaged over the patch
    weights = np.array([0.299, 0.587, 0.114])
    pixels = np.asarray(patch, dtype=np.float64)
    lum = float((pixels @ weights).mean())

    for _ in range(20):
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        color_lum = float(np.dot(color, weights))

        if abs(color_lum - lum) > 90:
            return color

    return (0, 0, 0) if lum > 127 else (255, 255, 255)


def _word_image(
    text: str,
    backgrounds: list[Image.Image],
    font: str,
    rng: random.Random,
    size: int = 512,
) -> Image.Image:
    """
    The word drawn at a random size and position on a random center-cropped background.
    """
    # Random background, center-cropped square and resized
    image = rng.choice(backgrounds)
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    image = image.crop(
        ((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)
    )
    image = image.resize((size, size), Image.LANCZOS)

    # Random font size and in-bounds position
    draw = ImageDraw.Draw(image)
    font_size = rng.randint(int(size * 0.10), int(size * 0.22))
    face = ImageFont.truetype(font, font_size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=face)
    tw, th = right - left, bottom - top
    margin = int(size * 0.02)
    x = rng.randint(margin - left, max(margin - left, size - tw - margin - left))
    y = rng.randint(margin - top, max(margin - top, size - th - margin - top))

    # Color contrasting with the region under the text
    box = (
        max(0, x + left),
        max(0, y + top),
        min(size, x + left + tw),
        min(size, y + top + th),
    )
    region = image.crop(box)
    fill = _contrasting_color(region, rng)
    draw.text((x, y), text, fill=fill, font=face)
    return image


@torch.no_grad()
def _head_probs(
    model: LlavaNextForConditionalGeneration,
    processor: LlavaNextProcessor,
    prompt: str,
    token_ids: list[int],
    images: list[Image.Image],
) -> torch.Tensor:
    """
    Per-head P(word) summed over a batch, (layers, heads).

    - o_proj input at the last position, split per head, through its O slice and the logit lens
    - prompt: chat-templated transcription text, shared by every image in the batch
    """
    cfg = model.config.text_config
    lm = model.model.language_model
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads

    # One prompt row per image
    inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Capture every layer's o_proj input at the last position
    captured = []
    hooks = [
        layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, args: captured.append(args[0][:, -1].clone())
        )
        for layer in lm.layers
    ]

    try:
        model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()

    # Per layer: head outputs -> residual space -> P(target token)
    batch = len(images)
    ids = torch.tensor(token_ids, device=model.device)
    rows = torch.arange(batch, device=model.device)
    out = torch.zeros(len(lm.layers), n_heads, device=model.device)

    for index, layer in enumerate(lm.layers):
        w_o = layer.self_attn.o_proj.weight
        w_o = w_o.view(w_o.shape[0], n_heads, head_dim)
        heads = captured[index].view(batch, n_heads, head_dim)
        projected = torch.einsum("bnh,dnh->bnd", heads, w_o)
        normed = lm.norm(projected)
        logits = model.lm_head(normed)
        probs = logits.float().softmax(-1)
        out[index] = probs[rows, :, ids].sum(0)

    return out


def score_ocr_heads(
    model_id: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
    *,
    n: int = 1024,
    batch_size: int = 4,
    seed: int = 177,
    dtype: str = "float16",
    font: str = "DejaVuSans.ttf",
) -> torch.Tensor:
    """
    Mean OCR score per attention head: how strongly each head writes the word shown in an image.

    - score: P(word) from the head's O-projected output at the last prompt position
    - loads its own full model; batch_size 16 OOMs the 7B in 46 GB

    Args:
        model_id: LLaVA-1.6 hub id or local directory.
        n: number of word images.
        batch_size: images per forward.
        seed: seeds a local RNG for word sampling and rendering.
        dtype: torch dtype name for the model weights.
        font: TrueType font for the words.

    Returns:
        (layers, heads) float32 CPU mean scores, usable as `head_scores`.

    Raises:
        ValueError: `n` not in [1, number of eligible words].
        OSError: `font` cannot be opened.
    """
    # Local RNG for word sampling and rendering; same draws as seeding the global one
    rng = random.Random(seed)

    # Words whose exact lowercase `▁word` token exists, 2-8 letters, as upstream
    processor = load_processor(model_id)
    vocab = word_vocabulary(processor.tokenizer)
    token_ids = vocab.token_ids.tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
    pairs = []

    for token, token_id, row in zip(tokens, token_ids, vocab.word_index.tolist()):
        word = token[1:]

        if word == vocab.words[row] and len(word) <= 8:
            pairs.append((word, token_id))

    # Check n and the font before the backgrounds and (7B) weights load
    if not 1 <= n <= len(pairs):
        raise ValueError(f"n must be in [1, {len(pairs)}], got {n}")

    ImageFont.truetype(font, 10)

    pairs = rng.sample(pairs, k=n)
    backgrounds = _load_backgrounds()

    # Full model: the lens needs every layer's heads
    model = _load_model(model_id, dtype)
    device = get_device()
    model.to(device).eval()

    # Transcription prompt with the answer prefilled; same text for every batch
    request = {"type": "text", "text": "Transcribe the word in this image."}
    messages = [{"role": "user", "content": [{"type": "image"}, request]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt = prompt + "Word:"

    # Accumulate per-head P(word) over batches
    cfg = model.config.text_config
    total = torch.zeros(
        cfg.num_hidden_layers, cfg.num_attention_heads, device=model.device
    )
    starts = range(0, n, batch_size)

    try:
        for start in tqdm(starts, desc="score OCR heads"):
            batch = pairs[start : start + batch_size]
            images = [_word_image(word, backgrounds, font, rng) for word, _ in batch]
            batch_ids = [token_id for _, token_id in batch]
            total += _head_probs(model, processor, prompt, batch_ids, images)
    finally:
        # Free the model's GPU memory, also when a batch fails
        del model
        pytorch_gc()

    # Mean on the CPU
    mean = total / n
    return mean.cpu()
