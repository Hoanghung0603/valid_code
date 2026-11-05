"""Shared text normalization, tokenization and LCS utilities.

Normalization follows the organizers' rule for chunk matching:
NFKC -> unescape HTML entities -> lowercase -> collapse whitespace -> strip.

Tokenization:
- "ws":   split on whitespace (our best guess of the official scorer; unconfirmed).
- "cjk":  whitespace split, but every CJK character becomes its own token.
          Used for extraction QC so Chinese text is measured meaningfully.
"""
from __future__ import annotations

import html
import re
import unicodedata

_WS = re.compile(r"\s+")
_CJK = re.compile(
    r"([㐀-䶿一-鿿豈-﫿　-〿＀-￯])"
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = html.unescape(text)
    text = text.lower()
    return _WS.sub(" ", text).strip()


def tokenize(text: str, mode: str = "ws", normalized: bool = False) -> list[str]:
    if not normalized:
        text = normalize(text)
    if mode == "ws":
        return text.split()
    if mode == "cjk":
        return _CJK.sub(r" \1 ", text).split()
    raise ValueError(f"unknown tokenizer mode: {mode}")


def lcs_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Bit-parallel algorithm (Crochemore et al. 2001): O(len(a) * len(b) / wordsize),
    fast enough for whole documents in pure Python.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a  # bit-vector over the longer sequence
    masks: dict[str, int] = {}
    for i, tok in enumerate(a):
        masks[tok] = masks.get(tok, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for tok in b:
        m = masks.get(tok)
        if m is None:
            continue
        u = v & m
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def overlap(candidate: list[str], reference: list[str]) -> float:
    """Official-style overlap: LCS(c, g) / |g|."""
    if not reference:
        return 0.0
    return lcs_len(candidate, reference) / len(reference)


def f_beta(p: float, r: float, beta: float = 2.0) -> float:
    b2 = beta * beta
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0
