import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dwx_p3 import (band_of, build_pairs, cmd_wincheck, idx_lookup, norm_id,  # noqa: E402
                    sample_windows)


def test_band_of_va_norm_id():
    assert [band_of(r) for r in (1, 30, 31, 60, 61, 100, 101, 500)] == \
           ["1-30", "1-30", "31-60", "31-60", "61-100", "61-100", "101+", "101+"]
    assert norm_id("12") == 12 and norm_id("abc") == "abc"


def test_idx_lookup_uu_tien_cache_dau():
    a = {"1": [[10, 5, 0.9], [11, None, 0.1]]}
    b = {"1": [[11, 7, 0.2], [12, 8, 0.3]]}
    assert idx_lookup([a, b]) == {"10": 5, "11": 7, "12": 8}


def _setup():
    doclist = {"1": [10, 11, 30, 40, 50], "2": [20]}
    pool = {"1": [[10, 1, 0.9], [11, 2, 0.8]] + [[i, i, 0.1] for i in range(100, 105)],
            "2": [[20, 1, 0.5]]}
    # top-100 pool của câu 1: 10, 11, 100..104 (7 doc); split câu 1: [11, 30]
    split_docs = {"1": [11, 30], "2": [20]}
    idx_of = {"10": 1, "11": 2, "20": 3, "30": 4, "40": 5, "50": 6}
    new_ids = {"1": {"30", "40"}, "2": set()}
    return doclist, pool, split_docs, idx_of, new_ids


def test_build_pairs_coverage_va_thong_ke():
    doclist, pool, split_docs, idx_of, new_ids = _setup()
    cands, cache, rep = build_pairs(doclist, pool, split_docs, idx_of, new_ids, top_n=100)
    # câu 1: 10 (top-100), 11 (top-100 + split), 30 (split) -> chỉ 40, 50 cần chấm; 40 NEW, 50 pool ngoài coverage
    assert cands["1"] == [[40, 5], [50, 6]]
    assert cache["1"] == [[40, 5, 0.0], [50, 6, 0.0]]
    assert "2" not in cands  # 20 đã có coverage
    assert rep["cap"] == 2 and rep["cap_NEW"] == 1 and rep["cap_pool_ngoai_coverage"] == 1
    assert rep["TB_cap_cau"] == 1.0 and rep["doc_duy_nhat"] == 2
    assert rep["cap_theo_dai_hang_doclist"]["1-30"] == 2  # hạng 4, 5 trong doclist
    assert rep["cap_thieu_idx_khong_export_duoc"] == 0


def test_build_pairs_thieu_idx():
    doclist, pool, split_docs, idx_of, new_ids = _setup()
    idx_of = {k: v for k, v in idx_of.items() if k != "50"}
    cands, _cache, rep = build_pairs(doclist, pool, split_docs, idx_of, new_ids, top_n=100)
    assert cands["1"] == [[40, 5]]
    assert rep["cap_thieu_idx_khong_export_duoc"] == 1


def test_sample_windows_lay_mau_co_text():
    docwin = {"1": {"10": [[0, 5, 1.5], [5, 9, 2.5]], "11": [[0, 3, 0.5]]}}
    docs = {"10": "0123456789"}
    qtext = {"1": "cau hoi"}
    rows = sample_windows(docwin, docs, qtext, n=10, seed=1)
    assert len(rows) == 2  # doc 11 không có text -> bỏ
    assert {r["start"] for r in rows} == {0, 5}
    assert all(r["query"] == "cau hoi" and r["window"] == docs["10"][r["start"]:r["end"]] for r in rows)
    assert all(r["ref_logit"] in (1.5, 2.5) for r in rows)
    rows2 = sample_windows(docwin, docs, qtext, n=1, seed=7)
    assert len(rows2) == 1


def test_wincheck_khop_va_sai(tmp_path):
    ref = [{"qid": 1, "doc_id": 10, "start": 0, "end": 5, "ref_logit": 1.5, "query": "q", "window": "w"},
           {"qid": 1, "doc_id": 10, "start": 5, "end": 9, "ref_logit": 2.5, "query": "q", "window": "w"}]
    rp, gp, op = tmp_path / "ref.json", tmp_path / "got.json", tmp_path / "rep.json"
    rp.write_text(json.dumps(ref), encoding="utf-8")
    got = [{**{k: r[k] for k in ("qid", "doc_id", "start", "end", "ref_logit")},
            "logit": r["ref_logit"] + 0.004} for r in ref]
    gp.write_text(json.dumps(got), encoding="utf-8")
    class A:
        ref = str(rp)
        got = str(gp)
        report = str(op)
        tol = 1e-2
    assert cmd_wincheck(A) == 0
    rep = json.loads(op.read_text(encoding="utf-8"))
    assert rep["ok"] and rep["so_cap"] == 2 and rep["max_sai"] <= 1e-2

    got[0]["logit"] = 9.0
    gp.write_text(json.dumps(got), encoding="utf-8")
    assert cmd_wincheck(A) == 2
    assert not json.loads(op.read_text(encoding="utf-8"))["ok"]

    gp.write_text(json.dumps(got[:1]), encoding="utf-8")
    assert cmd_wincheck(A) == 2  # thiếu 1 cặp
