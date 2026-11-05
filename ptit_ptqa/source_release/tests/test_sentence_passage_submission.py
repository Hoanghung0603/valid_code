import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_sentence_passage_submission import (
    MAX_CHARS,
    MIN_CHARS,
    make_passages,
    map_passage_to_chunk,
    sentence_ranges,
    tokenize,
)


def test_sentence_ranges_keep_exact_text_offsets():
    source = "Câu đầu tiên. Câu thứ hai!\n\nCâu thứ ba?"
    ranges = sentence_ranges(source)
    assert [source[start:end] for start, end in ranges] == [
        "Câu đầu tiên.", "Câu thứ hai!", "Câu thứ ba?"
    ]


def test_passages_are_nonempty_source_substrings_and_sentence_aligned():
    sentences = [f"Thông tin y khoa câu số {i} có đủ nội dung kiểm chứng." for i in range(40)]
    source = " ".join(sentences)
    passages = make_passages(source)
    assert passages
    assert all(len(text) >= MIN_CHARS for _, _, text in passages)
    assert all(len(text) <= MAX_CHARS for _, _, text in passages)
    assert all(source[start:end] == text for start, end, text in passages)
    assert all(end - start >= MIN_CHARS for start, end, _ in passages)
    assert all(passages[i][1] <= passages[i + 1][0] for i in range(len(passages) - 1))


def test_oversized_single_sentence_is_excluded_not_cut_mid_sentence():
    source = "Thông tin không có ranh giới câu " * 100
    assert len(source) > MAX_CHARS
    assert make_passages(source) == []


def test_packer_flushes_before_combination_crosses_maximum():
    first = "A" * 500 + "."
    second = "B" * 1100 + "."
    third = "C" * 500 + "."
    passages = make_passages(" ".join((first, second, third)))
    assert all(len(text) <= MAX_CHARS for _, _, text in passages)
    assert all(text in {first, second, third} or (first in text) != (third in text) for _, _, text in passages)


def test_tokenizer_handles_vietnamese_and_filters_common_pronouns():
    assert tokenize("Tôi có đau bụng khi mang thai") == ["đau", "bụng", "mang", "thai"]


def test_passage_maps_to_best_ranked_containing_chunk():
    spans = __import__("numpy").array([[11, 0, 500], [12, 300, 900]], dtype="int64")
    mapped = map_passage_to_chunk(
        42,
        [(350, 420, "evidence sentence")],
        spans,
        {11: 2, 12: 1},
    )
    assert mapped == [(42, 350, 420, "evidence sentence", 12)]
