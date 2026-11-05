import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dwx_p3 import cmd_pairs, merge_pairs, summarize_pairs  # noqa: E402


def test_merge_pairs_bo_trung_theo_thu_tu():
    cands, cache = {}, {}
    merge_pairs(cands, cache, {"1": [[10, 1], [20, 2]]}, {"1": [[10, 1, 0.0], [20, 2, 0.0]]})
    merge_pairs(cands, cache, {"1": [[20, 2], [30, 3]], "2": [[40, 4]]},
                {"1": [[20, 2, 0.0], [30, 3, 0.0]], "2": [[40, 4, 0.0]]})
    assert cands == {"1": [[10, 1], [20, 2], [30, 3]], "2": [[40, 4]]}
    assert cache["1"] == [[10, 1, 0.0], [20, 2, 0.0], [30, 3, 0.0]]


def test_summarize_pairs_dem_new():
    cands = {"1": [[10, 1], [30, 2]], "2": [[40, 3]]}
    cache = {}
    new_ids = {"1": {"30"}, "2": set()}
    rep = summarize_pairs(cands, cache, new_ids, n_cau=4)
    g = rep["gop"]
    assert g["cap"] == 3 and g["cap_NEW"] == 1 and g["TB_NEW_cau"] == 0.25
    assert g["cap_pool_ngoai_coverage"] == 2
    assert g["TB_cap_cau"] == 0.75 and g["doc_duy_nhat"] == 3
    assert g["min_cap"] == 1 and g["max_cap"] == 2 and g["cau_co_cap"] == 2


class _A:
    doclist = None
    pool = None
    split = None
    idx_from = []
    new = ""
    out_dir = None
    top_n = 100


def test_cmd_pairs_gop_hai_doclist(tmp_path):
    pool = {"1": [[10, 1, 0.9]]}  # top-100: doc 10
    split = [{"id": 1, "relevant_docs": [30], "relevant_chunks": []}]  # coverage: doc 30
    dl_a = {"1": [10, 40, 50]}  # 40, 50 thiếu coverage
    dl_b = {"1": [50, 60], "2": [20]}  # 50 trùng, 60 mới; câu 2 có coverage 20? không -> 20 cần
    idx_of = {"10": 1, "30": 2, "40": 3, "50": 4, "60": 5, "20": 6}
    for name, obj in (("pool.json", pool), ("split.json", split),
                      ("a.json", dl_a), ("b.json", dl_b), ("new.json", {"1": [40, 60]})):
        (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")
    # idx-from cache chứa idx cho các doc
    (tmp_path / "idx.json").write_text(
        json.dumps({"1": [[d, i, 0.0] for d, i in idx_of.items()]}), encoding="utf-8")
    a = _A()
    a.doclist = [str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    a.pool, a.split = str(tmp_path / "pool.json"), str(tmp_path / "split.json")
    a.idx_from = [str(tmp_path / "idx.json")]
    a.new = str(tmp_path / "new.json")
    a.out_dir = str(tmp_path / "out")
    assert cmd_pairs(a) == 0
    cands = json.loads((tmp_path / "out" / "cands_docwin_p3.json").read_text(encoding="utf-8"))
    # cau 1: 40, 50 (tu a) + 60 (tu b), bo trung 50; cau 2: 20
    assert cands == {"1": [[40, 3], [50, 4], [60, 5]], "2": [[20, 6]]}
    rep = json.loads((tmp_path / "out" / "dwx_p3_report.json").read_text(encoding="utf-8"))
    assert rep["gop"]["cap"] == 4 and rep["gop"]["cap_NEW"] == 2  # 40, 60 NEW (cau 1); 50, 20 khong
    assert rep["gop"]["TB_cap_cau"] == 2.0  # 4 cap / 2 cau
    assert len(rep["cac_doclist"]) == 2
    cache = json.loads((tmp_path / "out" / "cache_docwin_p3.json").read_text(encoding="utf-8"))
    assert all(len(h) == 3 and h[2] == 0.0 for rows in cache.values() for h in rows)
