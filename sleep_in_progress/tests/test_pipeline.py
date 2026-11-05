import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import validate
from pipeline import Retriever, build_submission

VI_DOCS = {
    1: ("Sốt xuất huyết", "Sốt xuất huyết dengue do muỗi vằn truyền. Bệnh nhân sốt cao, đau đầu.\n" * 10),
    2: ("Đau dạ dày", "Viêm loét dạ dày tá tràng gây đau thượng vị, ợ chua, buồn nôn.\n" * 10),
    3: ("Tiểu đường", "Đái tháo đường tuýp 2 cần kiểm soát đường huyết và insulin.\n" * 10),
}


def _docs():
    out = [{"id": i, "url": "u", "host": "h", "title": t, "date": "", "text": x, "lang": "vi"} for i, (t, x) in VI_DOCS.items()]
    out.append({"id": 9, "url": "u", "host": "h", "title": "", "date": "", "text": "五倍子适量，为细末。" * 30, "lang": "zh"})
    return out


def test_retrieve_returns_matching_doc_and_verbatim_chunks():
    r = Retriever(_docs(), chunk_units=20)
    res = r.retrieve("muỗi vằn gây sốt xuất huyết dengue")
    assert res["relevant_docs"][0] == 1
    assert res["relevant_chunks"], "phải có chunk"
    by_id = {d["id"]: d["text"] for d in r.docs}
    for c in res["relevant_chunks"]:
        assert c["doc_id"] in res["relevant_docs"]
        assert c["chunk_text"] in by_id[c["doc_id"]]


def test_zh_docs_excluded_and_unknown_query_empty():
    r = Retriever(_docs())
    assert all(d["id"] != 9 for d in r.docs)
    assert r.retrieve("zzzz qqqq") == {"relevant_docs": [], "relevant_chunks": []}


def test_rel_ratio_prunes_weak_docs():
    r = Retriever(_docs())
    strict = r.retrieve("sốt xuất huyết", rel_ratio=0.99)
    assert strict["relevant_docs"] == [1]


def test_submission_passes_validation():
    r = Retriever(_docs())
    sub = build_submission(r, [(10, "đau thượng vị ợ chua"), (11, "zzzz")])
    assert validate(sub, 2) == []
    assert sub[0]["relevant_docs"][0] == 2
    assert sub[1] == {"id": 11, "relevant_docs": [], "relevant_chunks": []}
