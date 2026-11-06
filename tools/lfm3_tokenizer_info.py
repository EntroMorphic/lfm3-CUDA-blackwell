"""Tokenizer facts the engine relies on, verified rather than assumed (mirrors lfm::tokenizer_token_count)."""
import json
from pathlib import Path


def tokenizer_token_count(checkpoint, model_vocabulary=None):
    """Number of ids the tokenizer defines; requires them to be exactly 0..count-1 (no gaps).

    The output layer may have more rows (LFM3-Omni-2B: 128000 rows, 125033 tokens); ids at or above
    the count decode to nothing, so token choice is bounded by it.
    """
    data = json.loads((Path(checkpoint) / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer model.vocab must be a nonempty object")
    ids = set(vocab.values()) | {token["id"] for token in data["added_tokens"]}
    if min(ids) < 0 or (model_vocabulary is not None and max(ids) >= model_vocabulary):
        raise ValueError("tokenizer id outside the model vocabulary")
    count = max(ids) + 1
    if len(ids) != count:
        raise ValueError("tokenizer ids are not contiguous")
    return count
