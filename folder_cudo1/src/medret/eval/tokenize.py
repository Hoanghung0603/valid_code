"""Tokenisers for chunk matching. The official scorer uses the BGE-M3 tokenizer."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from functools import lru_cache

TokenizeFn = Callable[[list[str]], list[Sequence[Hashable]]]  # batched: texts -> token sequences


@lru_cache(maxsize=4)
def _hf_tokenizer(name: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(name)
    tok.no_truncation()
    tok.no_padding()
    return tok


def hf_tokenize(name: str = "BAAI/bge-m3") -> TokenizeFn:
    """Token ids without special tokens (<s>, </s> would inflate every LCS by 2)."""
    tok = _hf_tokenizer(name)

    def fn(texts: list[str]) -> list[Sequence[Hashable]]:
        return [enc.ids for enc in tok.encode_batch(texts, add_special_tokens=False)]

    return fn


def whitespace_tokenize(texts: list[str]) -> list[Sequence[Hashable]]:
    """Offline fallback for unit tests and quick checks; scores differ from the official ones."""
    return [t.split() for t in texts]


def get_tokenizer(name: str) -> TokenizeFn:
    return whitespace_tokenize if name == "whitespace" else hf_tokenize(name)
