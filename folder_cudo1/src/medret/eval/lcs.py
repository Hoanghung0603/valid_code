"""Longest common subsequence length over token sequences, bit-parallel.

Crochemore et al. / Hyyro bit-vector recurrence with Python big ints: O(len(b) * len(a) / 64)
word operations. Fast enough to score whole documents against reference chunks.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


class LCSMatcher:
    """Precomputes match masks for `a` so it can be compared against many `b` sequences."""

    __slots__ = ("n", "_full", "_masks")

    def __init__(self, a: Sequence[Hashable]):
        self.n = len(a)
        self._full = (1 << self.n) - 1
        masks: dict[Hashable, int] = {}
        for i, tok in enumerate(a):
            masks[tok] = masks.get(tok, 0) | (1 << i)
        self._masks = masks

    def lcs(self, b: Sequence[Hashable]) -> int:
        if not self.n or not b:
            return 0
        v = self._full
        masks, full = self._masks, self._full
        for tok in b:
            m = masks.get(tok)
            if m is None:
                continue
            u = v & m
            v = ((v + u) | (v - u)) & full
        return self.n - v.bit_count()


def lcs_length(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    # Bits over the longer sequence, Python-level loop over the shorter one.
    if len(a) < len(b):
        a, b = b, a
    return LCSMatcher(a).lcs(b)
