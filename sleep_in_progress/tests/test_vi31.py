import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi31


def test_constants():
    assert vi31.MAX_ZIP == 99_000_000
    assert vi31.DELTA0 == 57 and vi31.SANITY_DELTA == 57
    assert vi31.OUT == vi31.ROOT / "data/sub/vi31"
    assert vi31.Q8_V31 == vi31.ROOT / "data/sub/diag31/q8b_v31.json"
    assert vi31.Q8_V29 == vi31.ROOT / "data/sub/diag29/in1/q8b_v29.json"
    assert vi31.NAME == "vi31"


def test_input_spec_chi_doi_q8b_va_cands():
    import vi30
    a = vi31.input_spec(vi31.Q8_V29, vi31.CHUNK30)
    b = vi31.input_spec(vi31.Q8_V31, vi31.CHUNK31)
    diff = [x[0] for x, y in zip(a, b) if x != y]
    assert diff == ["q8b_v31.json", "cands_v31.json.zst"]
    srcs = {k: s for (_, s, k2), k in zip(a, ["pool", "q8b", "dense", "hyde", "cands", "ref"])}
    assert srcs["pool"] == vi30.OUT / "cache_pool_v30.json"
    assert srcs["dense"] == vi30.OUT / "dense_v30_top200.json"
    assert srcs["hyde"] == vi30.OUT / "hyde_v30_top200.json"
    assert srcs["ref"] == vi30.IN29 / "ref_chunks_v29.json"
    assert srcs["q8b"] == vi30.IN29 / "q8b_v29.json"
    assert srcs["cands"] == vi30.OUT / "chunk_dw_u900_v30.json.zst"


def test_union_doclists():
    out = vi31.union_doclists({"1": [3, 5]}, {"1": ["5", 7], "2": [9]}, None)
    assert out == {"1": [3, 5, 7], "2": [9]}


def test_cap_moi_chi_doc_chua_phu():
    from vi24 import missing_docwin
    dl31 = {"1": [1, 2, 3, 4], "2": [5]}
    chunks30 = {"1": [[1, 0, 0, 0.1, "a"]]}
    covered = vi31.union_doclists({"1": [2]}, {"1": [3]}, {"2": ["5"]})
    idx = {str(i): 10 + i for i in range(10)}
    assert missing_docwin(dl31, chunks30, covered, idx) == {"1": [[4, 14]]}


def test_doclist_diff():
    rep = vi31.doclist_diff({"1": [1, 2, 3], "2": [4]}, {"1": [1, 2, 9], "2": [4]}, top=2)
    assert rep["so_cau_doi"] == 1 and rep["TB_doc_them_cau"] == 0.5
    assert rep["TB_doc_bot_cau"] == 0.5 and rep["TB_top2_trung"] == 1.5


def test_coverage_8b():
    rep = vi31.coverage_8b({"1": [[1, 0], [2, 0], [3, 0]]}, {"1": [[2, 1.0]]}, top=2)
    assert rep == {"top": 2, "cap": 2, "co_8b": 1, "ty_le": 0.5}
