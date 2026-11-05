"""Loại dòng lặp (menu, chân trang, lời từ chối trách nhiệm) theo từng host.

Một dòng xuất hiện ở > max_share số trang của cùng host (host có ít nhất min_docs trang mẫu) coi là boilerplate.
Chỉ bắt được dòng lặp NGUYÊN VĂN; danh sách gợi ý đổi theo từng trang (vd zhongyibaodian) thì không bắt được.

Lưu ý: text sau khi lọc có thể nối hai dòng vốn cách nhau bởi dòng boilerplate đã bị bỏ. Điểm chunk dùng LCS
nên chịu được chuyện này, nhưng chunk nộp bài khi đó không còn là một đoạn liền mạch tuyệt đối của trang gốc.
"""
import json
from collections import Counter, defaultdict
from typing import Iterable


def _norm(line: str) -> str:
    return " ".join(line.split())


class BoilerplateFilter:
    def __init__(self, min_docs: int = 50, max_share: float = 0.2, max_fit_docs: int = 3000):
        self.min_docs, self.max_share, self.max_fit_docs = min_docs, max_share, max_fit_docs
        self._lines: dict[str, set[str]] = {}

    def fit(self, docs: Iterable[dict]) -> "BoilerplateFilter":
        """docs: dict có 'host' và 'text'. Chỉ đếm tối đa max_fit_docs trang đầu mỗi host (giữ bộ nhớ nhỏ)."""
        df: dict[str, Counter] = defaultdict(Counter)
        n: Counter = Counter()
        for d in docs:
            host = d["host"]
            if n[host] >= self.max_fit_docs:
                continue
            n[host] += 1
            df[host].update({_norm(x) for x in d["text"].split("\n") if x.strip()})
        self._lines = {}
        for host, counter in df.items():
            if n[host] < self.min_docs:
                continue
            cut = self.max_share * n[host]
            bad = {line for line, c in counter.items() if c > cut}
            if bad:
                self._lines[host] = bad
        return self

    def to_json(self) -> str:
        return json.dumps({h: sorted(s) for h, s in self._lines.items()}, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "BoilerplateFilter":
        obj = cls()
        obj._lines = {h: set(v) for h, v in json.loads(data).items()}
        return obj

    def boilerplate_lines(self, host: str) -> set[str]:
        return self._lines.get(host, set())

    def clean(self, host: str, text: str) -> str:
        bad = self._lines.get(host)
        if not bad:
            return text
        return "\n".join(x for x in text.split("\n") if _norm(x) not in bad)
