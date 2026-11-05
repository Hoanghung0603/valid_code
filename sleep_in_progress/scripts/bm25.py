"""BM25 dựa trên bm25s (chỉ mục thưa, co giãn tới hàng triệu tài liệu), token hoá theo ngôn ngữ.

vi: chữ thường, NFKC, unigram + bigram âm tiết (tiếng Việt viết tách âm tiết nên bigram ~ từ ghép).
zh: bigram ký tự Hán + token chữ/số rời.
Chưa bỏ dấu tiếng Việt: query sai chính tả vẫn còn phần token đúng; xem xét thêm biến thể không dấu sau.
"""
import re
import unicodedata
from pathlib import Path

import bm25s
import numpy as np

_WORD = re.compile(r"[^\W_]+", re.U)
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str, lang: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    if lang == "zh":
        toks: list[str] = []
        pos = 0
        for m in _CJK.finditer(text):
            toks += _WORD.findall(text[pos:m.start()])
            run = m.group()
            toks += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
            pos = m.end()
        toks += _WORD.findall(text[pos:])
        return toks
    words = _WORD.findall(text)
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = 0
        self._r: bm25s.BM25 | None = None

    def build(self, token_lists: list[list[str]]) -> "BM25":
        self.n = len(token_lists)
        self._r = bm25s.BM25(k1=self.k1, b=self.b)
        if self.n:
            self._r.index(token_lists, show_progress=False)
        return self

    def search(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        """Trả về [(chỉ số tài liệu, điểm)] giảm dần, chỉ gồm điểm > 0."""
        if not self.n or not query_tokens:
            return []
        ids, scores = self._r.retrieve([query_tokens], k=min(k, self.n), show_progress=False)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if s > 0]

    def save(self, path: Path) -> None:
        self._r.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BM25":
        obj = cls()
        obj._r = bm25s.BM25.load(str(path))
        obj.n = int(obj._r.scores["num_docs"])
        return obj
