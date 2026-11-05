"""Cắt tài liệu thành chunk, luôn là đoạn trích NGUYÊN VĂN từ text gốc (luật thi cấm viết lại/sinh mới).

Đơn vị đo độ dài: 'vi' -> từ tách theo khoảng trắng (âm tiết); 'zh' -> ký tự không phải khoảng trắng.
Thuật toán: tách theo dòng -> dòng quá dài tách theo câu -> câu quá dài cắt cứng theo đơn vị
-> gộp tham lam các mảnh liền kề đến khi chạm max_units. Có thể lặp lại overlap_units ở đầu chunk kế.
"""
import re
from dataclasses import dataclass

_LINE = re.compile(r"[^\n]+")
# Dấu kết câu: '.', '!', '?' chỉ tính khi theo sau là khoảng trắng/hết chuỗi (tránh cắt "PGS.TS", "3.5"); dấu CJK luôn tính.
_SENTENCE = re.compile(r"(?:[^。！？；.!?;]|[.!?;]+(?!\s|$))+(?:[.!?;]+(?=\s|$)|[。！？；]+)?|[。！？；]+", re.S)
_WORD = re.compile(r"\S+")
_CHAR = re.compile(r"\S")


@dataclass(frozen=True)
class Chunk:
    start: int  # offset trong text gốc
    end: int
    text: str  # == text[start:end], đã strip
    units: int


def detect_lang(text: str, sample: int = 2000) -> str:
    """'zh' nếu >30% ký tự đầu là chữ Hán, ngược lại 'vi'."""
    head = text[:sample]
    if not head:
        return "vi"
    cjk = sum(1 for c in head if "\u4e00" <= c <= "\u9fff")
    return "zh" if cjk > len(head) * 0.3 else "vi"


def count_units(s: str, lang: str) -> int:
    return len(_CHAR.findall(s)) if lang == "zh" else len(_WORD.findall(s))


def _hard_split(text: str, start: int, end: int, limit: int, lang: str):
    """Cắt cứng đoạn [start,end) thành các mảnh ≤ limit đơn vị."""
    spans = list((_CHAR if lang == "zh" else _WORD).finditer(text, start, end))
    for i in range(0, len(spans), limit):
        group = spans[i:i + limit]
        yield group[0].start(), group[-1].end(), len(group)


def _pieces(text: str, limit: int, lang: str):
    """Sinh (start, end, units) của các mảnh không vượt limit, giữ thứ tự tài liệu."""
    for line in _LINE.finditer(text):
        ls, le = line.start(), line.end()
        n = count_units(text[ls:le], lang)
        if n == 0:
            continue
        if n <= limit:
            yield ls, le, n
            continue
        for sent in _SENTENCE.finditer(text, ls, le):
            ss, se = sent.start(), sent.end()
            m = count_units(text[ss:se], lang)
            if m == 0:
                continue
            if m <= limit:
                yield ss, se, m
            else:
                yield from _hard_split(text, ss, se, limit, lang)


def chunk_document(text: str, lang: str | None = None, max_units: int = 150, min_units: int = 20,
                   overlap_units: int = 0) -> list[Chunk]:
    """Cắt `text` thành chunk ~max_units đơn vị.

    min_units: chunk cuối ngắn hơn ngưỡng này được nối vào chunk trước (nếu không vượt 1.5*max_units),
    tránh sinh chunk vụn làm giảm Precision.
    """
    if max_units < 1:
        raise ValueError("max_units phải ≥ 1")
    if overlap_units >= max_units:
        raise ValueError("overlap_units phải < max_units")
    lang = lang or detect_lang(text)
    pieces = list(_pieces(text, max_units, lang))
    chunks: list[tuple[int, int, int]] = []  # (start, end, units)
    cur: list[tuple[int, int, int]] = []
    total = 0
    for p in pieces:
        if cur and total + p[2] > max_units:
            chunks.append((cur[0][0], cur[-1][1], total))
            # overlap: giữ lại các mảnh cuối có tổng ≤ overlap_units
            keep, acc = [], 0
            for q in reversed(cur):
                if acc + q[2] > overlap_units:
                    break
                keep.append(q)
                acc += q[2]
            cur, total = list(reversed(keep)), acc
        cur.append(p)
        total += p[2]
    if cur and (not chunks or cur[-1][1] > chunks[-1][1]):  # bỏ phần đuôi chỉ gồm overlap
        chunks.append((cur[0][0], cur[-1][1], total))

    if len(chunks) >= 2 and chunks[-1][2] < min_units:
        s, e, u = chunks[-1]
        ps, pe, pu = chunks[-2]
        merged_units = count_units(text[ps:e], lang)
        if merged_units <= max_units * 1.5:
            chunks[-2:] = [(ps, e, merged_units)]
    return [Chunk(s, e, text[s:e].strip(), u) for s, e, u in chunks]
