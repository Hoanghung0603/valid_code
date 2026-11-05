import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from evaluate import chunk_match, f2, lcs_len, normalize, score, tokenize, validate  # noqa: E402


def test_normalize():
    assert normalize("  Sỏi&nbsp;THẬN\n\t ") == "sỏi thận"
    assert normalize("ＡＢＣ") == "abc"  # NFKC full-width


def test_lcs_non_contiguous():
    assert lcs_len("a b c d".split(), "a x c d".split()) == 3
    assert lcs_len([], ["a"]) == 0


def test_f2():
    assert f2(0, 0) == 0
    assert abs(f2(0.5, 1.0) - 5 * 0.5 / 3) < 1e-9


def test_cjk_tokenizer():
    assert tokenize("慢性丙肝 abc", "cjk") == ["慢", "性", "丙", "肝", "abc"]
    assert tokenize("慢性丙肝 abc", "ws") == ["慢性丙肝", "abc"]


def _item(qid, docs, chunks):
    return {"id": qid, "relevant_docs": docs, "relevant_chunks": [{"doc_id": d, "chunk_text": t} for d, t in chunks]}


def test_validate():
    ok = [_item(1, [10], [(10, "x")])]
    assert validate(ok, n_queries=1) == []
    bad = [_item(1, [10], [(11, "x")])]
    assert any("không có trong relevant_docs" in e for e in validate(bad, n_queries=1))
    assert any("cần đúng" in e for e in validate(ok, n_queries=2))


def test_score_threshold():
    gold_text = "một hai ba bốn năm sáu bảy tám chín mười"
    gold = [_item(1, [10, 20], [(10, gold_text)])]
    # 4/10 token chung -> đạt ngưỡng 40%; doc 20 bỏ sót
    sub = [_item(1, [10], [(10, "một hai ba bốn khác")])]
    r = score(sub, gold)
    assert r["doc_p"] == 1.0 and r["doc_r"] == 0.5
    assert r["chunk_p"] == 1.0 and r["chunk_r"] == 1.0
    # 3/10 -> trượt
    r2 = score([_item(1, [10], [(10, "một hai ba")])], gold)
    assert r2["chunk_p"] == 0.0 and r2["chunk_f2"] == 0.0
    # sai doc_id thì không tính dù trùng text
    r3 = score([_item(1, [20], [(20, gold_text)])], gold)
    assert r3["chunk_p"] == 0.0


G10 = " ".join(f"g{i}" for i in range(10))  # 10 token vàng


def _gold(gtxt=G10):
    return [_item(1, [10], [(10, gtxt)])]


def _sub(ctxt):
    return [_item(1, [10], [(10, ctxt)])]


def _hit(ctxt, gtxt, rule):
    r = score(_sub(ctxt), _gold(gtxt), chunk_rule=rule)
    return r["chunk_p"] == 1.0


def test_chunk_rule_pred_3x_contains_gold():
    # chunk nộp dài gấp 3 lần chunk vàng, chứa trọn chunk vàng: LCS/|g|=1, LCS/|c|=10/30
    ctxt = G10 + " " + " ".join(f"x{i}" for i in range(20))
    r_old = score(_sub(ctxt), _gold(), chunk_rule="gold")
    r_new = score(_sub(ctxt), _gold(), chunk_rule="both")
    assert r_old["chunk_p"] == 1.0 and r_old["chunk_r"] == 1.0
    assert r_new["chunk_p"] == 0.0 and r_new["chunk_r"] == 0.0


def test_chunk_rule_exact_match():
    for rule in ("gold", "both"):
        r = score(_sub(G10), _gold(), chunk_rule=rule)
        assert r["chunk_p"] == 1.0 and r["chunk_r"] == 1.0 and r["chunk_f2"] == 1.0


def test_chunk_rule_pred_2x_contains_gold():
    # chứa trọn vàng trong chunk gấp đôi: LCS/|c| = 10/20 = 0.5 -> đúng cả hai luật
    ctxt = G10 + " " + " ".join(f"x{i}" for i in range(10))
    for rule in ("gold", "both"):
        r = score(_sub(ctxt), _gold(), chunk_rule=rule)
        assert r["chunk_p"] == 1.0 and r["chunk_r"] == 1.0


def test_chunk_rule_boundary_04():
    # LCS/|c| đúng biên 0.4: vàng 10 token chứa trong nộp 25 token (10/25=0.4) -> đúng cả hai
    c25 = G10 + " " + " ".join(f"x{i}" for i in range(15))
    c26 = G10 + " " + " ".join(f"x{i}" for i in range(16))  # 10/26 < 0.4
    assert _hit(c25, G10, "gold") and _hit(c25, G10, "both")
    assert _hit(c26, G10, "gold") and not _hit(c26, G10, "both")
    # LCS/|g| đúng biên 0.4: nộp 4/10 token của vàng -> đúng cả hai; 3/10 -> trượt cả hai
    assert _hit("g0 g1 g2 g3", G10, "gold") and _hit("g0 g1 g2 g3", G10, "both")
    assert not _hit("g0 g1 g2", G10, "gold") and not _hit("g0 g1 g2", G10, "both")
    # LCS/|c| biên với LCS < |c|: 8 token vàng + 12 rác = 20 (8/20=0.4) -> đúng; 8/21 -> trượt both
    c8_20 = "g0 g1 g2 g3 g4 g5 g6 g7 " + " ".join(f"y{i}" for i in range(12))
    c8_21 = "g0 g1 g2 g3 g4 g5 g6 g7 " + " ".join(f"y{i}" for i in range(13))
    assert _hit(c8_20, G10, "both")
    assert _hit(c8_21, G10, "gold") and not _hit(c8_21, G10, "both")


def test_chunk_rule_empty_chunk():
    # chunk rỗng không khớp theo cả hai luật
    assert not _hit("", G10, "gold") and not _hit("", G10, "both")
    assert not _hit(G10, "", "gold") and not _hit(G10, "", "both")
    assert not chunk_match([], G10.split(), "both")
    assert not chunk_match(G10.split(), [], "gold")


def test_chunk_rule_recall_new():
    # recall: 1 chunk vàng khớp tuyệt đối, 1 chunk vàng bị gói trong chunk nộp gấp 3 lần
    g2 = " ".join(f"h{i}" for i in range(10))
    gold = [_item(1, [10], [(10, G10), (10, g2)])]
    c_long = g2 + " " + " ".join(f"x{i}" for i in range(20))
    sub = [_item(1, [10], [(10, G10), (10, c_long)])]
    r_old = score(sub, gold, chunk_rule="gold")
    r_new = score(sub, gold, chunk_rule="both")
    assert r_old["chunk_r"] == 1.0 and r_old["chunk_p"] == 1.0
    assert r_new["chunk_r"] == 0.5 and r_new["chunk_p"] == 0.5
