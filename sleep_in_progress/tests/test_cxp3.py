import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import cxp3  # noqa: E402
from cx110 import norm_pairs  # noqa: E402
from cx93 import apply_bonus, filter_base, select_q  # noqa: E402
from cxp3 import (build_sub, check_zip, order_q2, parse_doclists, prep_queries,  # noqa: E402
                  repro, run, swapdoc)
from diag14 import order_q, strip_meta  # noqa: E402
from zip_submission import pack  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cx93 import BASE1, FU1, FU2, _rows1, _rows2  # noqa: E402


def _rows3():
    # doc 10 (cũ), doc 30 NEW (không có trong fused cũ, có trong fused mới)
    return [[10, 1, 0, 3.0, "A"], [30, 2, 1, 2.5, "N"], [10, 1, 2, 0.5, "C"]]


FU3 = [[10, 0, 1.0]]
FU3_NEW = [[30, 0, 2.0], [10, 0, 0.1]]


def test_order_q2_khong_fused_new_giong_order_q():
    for rows, fu in ((_rows1(), FU1), (_rows2(), FU2), (_rows3(), FU3)):
        assert order_q2(rows, fu, None, 4.0) == order_q(rows, fu, 4.0)
        assert order_q2(rows, fu, [], 4.0) == order_q(rows, fu, 4.0)


def test_order_q2_zd_doc_new_dung_cache_moi():
    got = order_q2(_rows3(), FU3, FU3_NEW, 4.0)
    by = {c["chunk_text"]: c for c in got}
    texts = [c["chunk_text"] for c in got]
    # fused cũ chỉ có doc 10 (một hit -> z = 0, zmin = 0); doc NEW 30 lấy z của fused mới:
    # z(2.0 trên [2.0, 0.1]) = +1 -> s = zc + 4*1 -> N vươn lên đứng đầu (không có fused_new thì A đứng đầu)
    assert by["N"]["zd"] == pytest.approx(1.0)
    assert by["A"]["zd"] == pytest.approx(0.0)
    assert by["C"]["zd"] == pytest.approx(0.0)
    assert texts[0] == "N"
    assert [c["chunk_text"] for c in order_q2(_rows3(), FU3, None, 4.0)][0] == "A"


def test_order_q2_doc_thieu_cung_fused_lay_zmin():
    rows = [[99, 1, 0, 1.0, "X"]] + _rows3()
    got = order_q2(rows, FU3, FU3_NEW, 4.0)
    by = {c["chunk_text"]: c for c in got}
    assert by["X"]["zd"] == pytest.approx(0.0)  # zmin của fused cũ (doc 10 z = 0)


def test_parse_doclists():
    assert parse_doclists(["a=b.json", "c=d.json"]) == {"a": "b.json", "c": "d.json"}
    with pytest.raises(ValueError):
        parse_doclists(["sai"])


def _ref_cx130(rows, fu, base, k):
    """Bài cx130cov giả lập: k' = min(k + 2, |C_q|) chunk của công thức (cho repro)."""
    ob = apply_bonus(filter_base(order_q(rows, fu, 4.0), base), 1.0)
    return [{"id": 1, "relevant_docs": sorted(int(d) for d in base),
             "relevant_chunks": strip_meta(ob, min(k + 2, len(ob)))}]


def test_swapdoc_noi_doc_thieu_va_giu_chunk(tmp_path):
    ref = _ref_cx130(_rows3(), FU3, {"10", "30"}, 2)
    cx = tmp_path / "cx.json"
    cx.write_text(json.dumps(ref), encoding="utf-8")
    dl = tmp_path / "dl.json"
    dl.write_text(json.dumps({"1": [30]}), encoding="utf-8")  # thiếu doc 10 (nguồn chunk)
    out = tmp_path / "out"
    rep = swapdoc(str(cx), {"sub_a": str(dl)}, str(out))
    sub = json.loads((out / "sub_a.json").read_text(encoding="utf-8"))
    assert sub[0]["relevant_docs"] == [30, 10]  # doclist trước, doc chunk thiếu nối cuối
    assert sub[0]["relevant_chunks"] == ref[0]["relevant_chunks"]
    st = rep["bai"]["sub_a"]
    assert st["TB_doc_noi_cau"] == 1.0 and st["cau_co_doc_noi"] == 1
    assert st["TB_doc_cau"] == 2.0 and st["chunk_giong_cx130cov"]


def test_swapdoc_khong_noi_khi_da_day_du(tmp_path):
    ref = _ref_cx130(_rows3(), FU3, {"10", "30"}, 2)
    cx = tmp_path / "cx.json"
    cx.write_text(json.dumps(ref), encoding="utf-8")
    dl = tmp_path / "dl.json"
    dl.write_text(json.dumps({"1": [30, 10, 11]}), encoding="utf-8")
    rep = swapdoc(str(cx), {"sub_b": str(dl)}, str(tmp_path / "out"))
    sub = json.loads((tmp_path / "out" / "sub_b.json").read_text(encoding="utf-8"))
    assert sub[0]["relevant_docs"] == [30, 10, 11]
    assert rep["bai"]["sub_b"]["TB_doc_noi_cau"] == 0.0


def _files3(tmp_path, extra_rows=None):
    rows = extra_rows if extra_rows is not None else _rows3()
    files = {"cands.json": {"1": rows}, "fused.json": {"1": FU3}, "fused_new.json": {"1": FU3_NEW},
             "ref120.json": [{"id": 1, "relevant_docs": [], "relevant_chunks":
                              [{"doc_id": 10, "chunk_text": "x"}] * 2}],
             "new.json": {"1": [30]}}
    for name, obj in files.items():
        (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")
    dl1 = tmp_path / "dl_full.json"
    dl1.write_text(json.dumps({"1": [10, 30]}), encoding="utf-8")
    dl2 = tmp_path / "dl_old.json"
    dl2.write_text(json.dumps({"1": [10]}), encoding="utf-8")
    return {**{k: str(tmp_path / k) for k in files},
            "dl_full": str(dl1), "dl_old": str(dl2)}


def test_run_end_to_end_hai_doclist(tmp_path):
    p = _files3(tmp_path)
    out = tmp_path / "out"
    rep = run(p["cands.json"], {"sub_full": p["dl_full"], "sub_old": p["dl_old"]},
              p["fused.json"], p["fused_new.json"], p["ref120.json"], str(out),
              new_cands_path=p["new.json"], delta=3)
    st = rep["bai"]["sub_full"]
    assert st["kprime"] == {"TB": 3, "min": 3, "max": 3, "tong": 3}  # min(k_q 2+3, |C_q| 3) = 3
    assert st["TB_doc_cau"] == 2.0 and st["TB_doc_co_chunk_cau"] == 2.0
    assert st["TB_chunk_tu_doc_NEW_cau"] > 0  # chunk của doc 30 (NEW)
    sub = json.loads((out / "sub_full.json").read_text(encoding="utf-8"))
    assert sub[0]["relevant_docs"] == [10, 30]
    assert {str(c["doc_id"]) for c in sub[0]["relevant_chunks"]} <= {"10", "30"}
    st2 = rep["bai"]["sub_old"]
    sub2 = json.loads((out / "sub_old.json").read_text(encoding="utf-8"))
    assert sub2[0]["relevant_docs"] == [10]
    assert {str(c["doc_id"]) for c in sub2[0]["relevant_chunks"]} == {"10"}
    assert st2["TB_doc_co_chunk_cau"] == 1.0 and st2["TB_chunk_tu_doc_NEW_cau"] == 0.0
    assert rep["bai"]["sub_full"]["delta"]["dung"] == 3
    assert (out / "report_cxp3.json").exists()


def test_run_giam_delta_khi_uoc_zip_vuot(tmp_path):
    p = _files3(tmp_path)
    out = tmp_path / "out"
    rep = run(p["cands.json"], {"sub_full": p["dl_full"]}, p["fused.json"], p["fused_new.json"],
              p["ref120.json"], str(out), new_cands_path=p["new.json"],
              delta_start=3, max_zip=1)
    st = rep["bai"]["sub_full"]
    assert st["delta"]["dung"] == 0  # ngân sách 1 byte -> giảm hết
    assert not st["uoc_zip"]["duoi_gioi_han"]
    sub = json.loads((out / "sub_full.json").read_text(encoding="utf-8"))
    assert len(sub[0]["relevant_chunks"]) == 2  # min(k_q + 0, |C_q|) = 2


def test_repro_tai_lap_cx130(tmp_path):
    rows = _rows3()
    fu = {"1": FU3}
    ref130 = _ref_cx130(rows, FU3, {"10", "30"}, 2)
    p = _files3(tmp_path, extra_rows=rows)
    ref = tmp_path / "ref130.json"
    ref.write_text(json.dumps(ref130), encoding="utf-8")
    rp = tmp_path / "repro.json"
    rep = repro(p["cands.json"], str(ref), p["fused.json"], p["ref120.json"], str(rp),
                delta=2, min_match=0.99)
    assert rep["ok"] and rep["ty_le_cau_khop"] == 1.0
    bad = json.loads(json.dumps(ref130))
    bad[0]["relevant_chunks"] = bad[0]["relevant_chunks"][:1]
    ref.write_text(json.dumps(bad), encoding="utf-8")
    rep2 = repro(p["cands.json"], str(ref), p["fused.json"], p["ref120.json"], str(rp),
                 delta=2, min_match=0.99)
    assert not rep2["ok"] and rep2["ty_le_cau_khop"] == 0.0


def test_prep_queries_va_build_sub_assert_ngoai_doclist():
    doclist = {"1": [10]}
    kq = {"1": 2}
    prep = prep_queries({"1": _rows3()}, doclist, {"1": FU3}, {"1": FU3_NEW}, kq, 4.0, 1.0)
    assert prep["1"]["kq"] == 2 and prep["1"]["cap"] == 2  # chỉ doc 10: A, C
    sub, st = build_sub(["1"], doclist, prep, {"1": {"30"}}, 0)
    assert [c["chunk_text"] for c in sub[0]["relevant_chunks"]] == ["A", "C"]
    assert st["TB_chunk_tu_doc_NEW_cau"] == 0.0
    prep_bad = {"1": {"kq": 2, "cap": 2, "ob": [{"doc_id": 30, "chunk_text": "N"}]}}
    with pytest.raises(AssertionError):
        build_sub(["1"], doclist, prep_bad, {}, 1)


def test_check_zip_ok_vuot_va_khac(tmp_path):
    name = "sub_x"
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps([{"id": 1, "relevant_docs": [1], "relevant_chunks": []}]), encoding="utf-8")
    z = tmp_path / f"{name}.zip"
    pack(p, z, method="lzma")
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps({"bai": {name: {"delta": {"dung": 5, "thu": []}}}}), encoding="utf-8")
    assert check_zip(str(p), str(z), str(rp), name, max_zip=10 ** 9) == 0
    rep = json.loads(rp.read_text(encoding="utf-8"))
    assert rep["zip"][name]["ok"] and rep["zip"][name]["bang_goc_bytes"]
    assert len(rep["zip"][name]["sha256"]) == 64

    assert check_zip(str(p), str(z), str(rp), name, max_zip=1) == 3
    assert (tmp_path / f"next_delta_{name}.txt").read_text(encoding="utf-8") == "--delta 4"

    rp.write_text(json.dumps({"bai": {name: {"delta": {"dung": 0, "thu": []}}}}), encoding="utf-8")
    assert check_zip(str(p), str(z), str(rp), name, max_zip=1) == 4

    p.write_text(json.dumps([{"id": 1, "relevant_docs": [2], "relevant_chunks": []}]), encoding="utf-8")
    assert check_zip(str(p), str(z), str(rp), name, max_zip=10 ** 9) == 1
