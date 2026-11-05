import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bm25 import BM25, tokenize
from boilerplate import BoilerplateFilter


def _docs(n, host="h.cn", menu="首页\n检 索", body="正文{i}"):
    return [{"host": host, "text": f"{menu}\n{body.format(i=i)}"} for i in range(n)]


def test_boilerplate_removes_repeated_lines_only_for_big_hosts():
    f = BoilerplateFilter(min_docs=50, max_share=0.2).fit(_docs(60))
    assert f.clean("h.cn", "首页\n正文1\n检 索") == "正文1"
    assert f.boilerplate_lines("h.cn") == {"首页", "检 索"}
    # host ít mẫu -> không học gì
    g = BoilerplateFilter(min_docs=50).fit(_docs(10))
    assert g.clean("h.cn", "首页\n正文") == "首页\n正文"
    # host lạ -> giữ nguyên
    assert f.clean("other", "首页") == "首页"


def test_boilerplate_keeps_lines_below_threshold():
    docs = _docs(100, menu="首页")
    for d in docs[:10]:  # 10% số trang có thêm dòng này -> dưới ngưỡng 20%
        d["text"] += "\n偶尔出现"
    f = BoilerplateFilter(min_docs=50, max_share=0.2).fit(docs)
    assert "偶尔出现" in f.clean("h.cn", "偶尔出现\n正文")
    assert "首页" not in f.clean("h.cn", "首页\n正文")


def test_boilerplate_whitespace_insensitive():
    f = BoilerplateFilter(min_docs=50).fit(_docs(60, menu="检 索"))
    assert f.clean("h.cn", "检   索\nabc") == "abc"


def test_tokenize_vi_has_unigram_and_bigram():
    t = tokenize("Sốt xuất huyết", "vi")
    assert t == ["sốt", "xuất", "huyết", "sốt_xuất", "xuất_huyết"]


def test_tokenize_zh_bigrams_and_ascii():
    assert tokenize("五倍子膏 B12", "zh") == ["五倍", "倍子", "子膏", "b12"]
    assert tokenize("膏", "zh") == ["膏"]


def test_tokenize_nfkc_and_case():
    assert tokenize("ＡＢＣ", "vi") == ["abc"]


def test_bm25_ranks_relevant_first():
    corpus = ["sốt xuất huyết dengue muỗi vằn", "đau dạ dày viêm loét", "tiểu đường tuýp 2 insulin", "sốt cao co giật trẻ em"]
    bm = BM25().build([tokenize(c, "vi") for c in corpus])
    hits = bm.search(tokenize("muỗi vằn gây sốt xuất huyết", "vi"), k=3)
    assert hits[0][0] == 0
    assert all(s > 0 for _, s in hits)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_bm25_unknown_terms_and_k_larger_than_n():
    bm = BM25().build([tokenize("abc def", "vi"), tokenize("ghi", "vi")])
    assert bm.search(tokenize("zzz", "vi"), k=5) == []
    assert len(bm.search(tokenize("abc ghi", "vi"), k=50)) == 2


def test_bm25_length_normalisation():
    short, long_ = "ung thư phổi", "ung thư phổi " + "từ khác " * 200
    bm = BM25().build([tokenize(long_, "vi"), tokenize(short, "vi")])
    assert bm.search(tokenize("ung thư phổi", "vi"), k=2)[0][0] == 1
