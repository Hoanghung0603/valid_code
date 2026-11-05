import pytest

from medret.eval.metric import evaluate, fbeta
from medret.eval.normalize import normalize_text
from medret.eval.tokenize import whitespace_tokenize
from medret.schemas import ChunkRef, QueryResult

A = "sỏi thận gây tắc nghẽn đường tiết niệu cần được xử trí sớm"
B = "tán sỏi ngoài cơ thể là lựa chọn cho sỏi nhỏ hơn hai centimet"
C = "kidney stones obstructing the ureter require urgent drainage"
NOISE = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do"


def run(pred, gold):
    return evaluate(pred, gold, whitespace_tokenize)


def q(qid, docs, chunks=()):
    return QueryResult(qid, list(docs), [ChunkRef(d, t) for d, t in chunks])


def test_fbeta():
    assert fbeta(1, 1) == 1
    assert fbeta(0, 0) == 0
    assert fbeta(0.5, 1) == pytest.approx(5 * 0.5 / (4 * 0.5 + 1))


def test_doc_level_one_gold_k_returned():
    gold = [q(1, ["10"])]
    for k, expected in [(1, 1.0), (2, 5 / 6), (5, 5 / 9)]:
        pred = [q(1, ["10"] + [f"x{i}" for i in range(k - 1)])]
        assert run(pred, gold).doc_f2 == pytest.approx(expected)


def test_int_and_str_doc_ids_are_equivalent():
    gold = [QueryResult.from_dict({"id": 1, "relevant_docs": [10]})]
    pred = [QueryResult.from_dict({"id": 1, "relevant_docs": ["10"]})]
    assert run(pred, gold).doc_f2 == 1


def test_whole_document_chunk_covered_every_reference_under_the_old_rule():
    # Before 2026-10-24 the only denominator was the reference length, so a long prediction was not penalised.
    gold = [q(1, ["10"], [("10", A), ("10", B), ("10", C)])]
    whole_doc = " ".join([NOISE, A, NOISE, B, C, NOISE])
    r = evaluate([q(1, ["10"], [("10", whole_doc)])], gold, whitespace_tokenize, pred_threshold=None)
    assert r.chunk_f2 == 1


def test_prediction_longer_than_2_5x_the_reference_does_not_match():
    gold = [q(1, ["10"], [("10", A)])]
    assert run([q(1, ["10"], [("10", " ".join([A, NOISE]))])], gold).chunk_f2 == 1  # 23 tokens vs 11: 0.48
    long = " ".join([NOISE, A, NOISE])  # 31 tokens: LCS 11 / 31 = 0.35 < 0.4
    assert run([q(1, ["10"], [("10", long)])], gold).chunk_f2 == 0


def test_chunk_from_wrong_doc_is_irrelevant():
    gold = [q(1, ["10"], [("10", A)])]
    r = run([q(1, ["10"], [("99", A)])], gold)
    assert r.per_query[0].chunk.precision == 0
    assert r.per_query[0].chunk.recall == 0


def test_relevance_threshold_is_40_percent_of_reference():
    ref = " ".join(f"t{i}" for i in range(10))
    gold = [q(1, ["10"], [("10", ref)])]
    four = run([q(1, ["10"], [("10", "t0 t1 t2 t3 zz")])], gold)
    three = run([q(1, ["10"], [("10", "t0 t1 t2 zz zz")])], gold)
    assert four.per_query[0].chunk.recall == 1
    assert three.per_query[0].chunk.recall == 0


def test_near_duplicates_count_separately_by_default():
    gold = [q(1, ["10"], [("10", A)])]
    near_dup = A + " ạ"  # 13 vs 14 tokens -> LCS/union = 13/14
    preds = [q(1, ["10"], [("10", A), ("10", near_dup), ("10", NOISE)])]
    s = run(preds, gold).per_query[0]
    assert s.n_pred_chunks == 3 and s.chunk.precision == pytest.approx(2 / 3)
    merged = evaluate(preds, gold, whitespace_tokenize, near_duplicate_threshold=0.8).per_query[0]  # old spec
    assert merged.n_pred_chunks == 2 and merged.chunk.precision == 0.5


def test_macro_skips_queries_without_positives():
    gold = [q(1, ["10"], [("10", A)]), q(2, [])]
    pred = [q(1, ["10"], [("10", A)]), q(2, ["7"], [("7", NOISE)])]
    r = run(pred, gold)
    assert (r.n_doc_queries, r.n_chunk_queries) == (1, 1)
    assert r.final == 1


def test_missing_prediction_scores_zero():
    gold = [q(1, ["10"], [("10", A)]), q(2, ["11"], [("11", B)])]
    r = run([q(1, ["10"], [("10", A)])], gold)
    assert r.doc_f2 == 0.5 and r.chunk_f2 == 0.5


def test_normalize_text():
    # NFKC folds the full-width comma; entities are decoded; tags and curly quotes are left as they are
    assert normalize_text("<p>Sỏi&nbsp;Thận，“tắc”  nghẽn</p>") == "<p>sỏi thận,“tắc” nghẽn</p>"
    # Decomposed Vietnamese diacritics compose to the same string as precomposed ones.
    assert normalize_text("thế") == normalize_text("thế")
