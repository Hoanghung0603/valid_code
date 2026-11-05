import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from chunking import chunk_document, count_units, detect_lang
from corpus import iter_docs
from shards import ShardWriter

VI = ("Sốt xuất huyết là bệnh truyền nhiễm do virus dengue gây ra. Bệnh lây qua muỗi vằn đốt người. "
      "Triệu chứng gồm sốt cao, đau đầu, đau cơ và xuất huyết dưới da.\n") * 12
ZH = "五倍子适量，为细末，米醋熬膏。先抓破患处，后将膏敷上，治顽癣。" * 30


def test_detect_lang():
    assert detect_lang(VI) == "vi"
    assert detect_lang(ZH) == "zh"
    assert detect_lang("") == "vi"


@pytest.mark.parametrize("text,lang", [(VI, "vi"), (ZH, "zh")])
def test_chunks_are_verbatim_and_bounded(text, lang):
    chunks = chunk_document(text, lang, max_units=40, min_units=5)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text == text[c.start:c.end].strip()  # nguyên văn
        assert c.text in text
        assert 0 < c.units <= 40 * 1.5
        assert count_units(c.text, lang) == c.units


@pytest.mark.parametrize("text,lang", [(VI, "vi"), (ZH, "zh")])
def test_chunks_cover_all_content_without_overlap(text, lang):
    chunks = chunk_document(text, lang, max_units=40, min_units=5)
    # không chồng lắp, đúng thứ tự
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start
    joined = "".join("".join(c.text.split()) for c in chunks)
    assert joined == "".join(text.split())  # không mất ký tự nào


def test_overlap_repeats_tail_pieces():
    text = "\n".join(f"Dòng số {i} có vài từ." for i in range(20))
    chunks = chunk_document(text, "vi", max_units=20, min_units=1, overlap_units=10)
    assert len(chunks) > 2
    assert any(b.start < a.end for a, b in zip(chunks, chunks[1:]))
    for c in chunks:
        assert c.units <= 20


def test_long_line_split_by_sentence_then_hard():
    long_sentence = " ".join(f"từ{i}" for i in range(500))  # một câu không dấu chấm
    chunks = chunk_document(long_sentence, "vi", max_units=100, min_units=5)
    assert len(chunks) == 5
    assert all(c.units == 100 for c in chunks)


def test_abbreviation_and_decimal_not_split():
    text = "PGS.TS Nguyễn Văn A cho biết tỷ lệ là 3.5 phần trăm. Câu thứ hai ngắn."
    chunks = chunk_document(text, "vi", max_units=8, min_units=1)
    assert any("PGS.TS" in c.text for c in chunks)
    assert any("3.5" in c.text for c in chunks)


def test_tiny_tail_merged():
    text = "\n".join(["một hai ba bốn năm sáu bảy tám chín mười"] * 3 + ["cuối"])
    chunks = chunk_document(text, "vi", max_units=20, min_units=5)
    assert chunks[-1].text.endswith("cuối")
    assert all(c.units >= 5 for c in chunks)


def test_empty_and_whitespace():
    assert chunk_document("", "vi") == []
    assert chunk_document("  \n \n", "vi") == []


def test_invalid_args():
    with pytest.raises(ValueError):
        chunk_document("abc", "vi", max_units=0)
    with pytest.raises(ValueError):
        chunk_document("abc", "vi", max_units=10, overlap_units=10)


def _write(dir_, recs):
    w = ShardWriter(dir_, "crawl")
    for r in recs:
        w.write(r)
    w.close()


def test_iter_docs_filters_and_dedups(tmp_path):
    base = {"url": "u", "host": "h", "title": "t", "date": "", "ts": 0}
    _write(tmp_path, [
        {**base, "id": 1, "status": 200, "text": VI},
        {**base, "id": 1, "status": 200, "text": VI},        # trùng id
        {**base, "id": 2, "status": 200, "text": ZH},
        {**base, "id": 3, "status": 404, "text": VI},        # không phải 200
        {**base, "id": 4, "status": 200, "text": "ngắn"},     # quá ngắn
    ])
    docs = list(iter_docs(tmp_path))
    assert [d["id"] for d in docs] == [1, 2]
    assert [d["lang"] for d in docs] == ["vi", "zh"]
    assert [d["id"] for d in iter_docs(tmp_path, langs={"zh"})] == [2]
    assert len(list(iter_docs(tmp_path, limit=1))) == 1
