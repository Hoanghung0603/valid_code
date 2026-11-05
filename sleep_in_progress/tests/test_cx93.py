import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunk_docfuse import pick_q  # noqa: E402
from cx93 import (aggregate, apply_bonus, build_subs, dup_text_stats, filter_base,  # noqa: E402
                  pair_set, repro_check, run, select_q, select_q_literal, stats_q, verify_zip_full)
from diag14 import build_sub, order_q, strip_meta  # noqa: E402
from zip_submission import pack  # noqa: E402


def _rows1():
    # doc 10, 11 thuộc doc nền; doc 12 ngoài doc nền; text "T" trùng giữa doc 11 và doc 12
    return [[10, 1, 0, 3.0, "A"], [11, 2, 1, 2.2, "T"], [12, 3, 2, 1.5, "T"],
            [11, 2, 3, 2.0, "B"], [10, 1, 4, 0.5, "C"]]


def _rows2():
    return [[20, 1, 0, 1.0, "P"], [21, 2, 1, 0.9, "Q"], [22, 3, 2, 0.8, "R"],
            [20, 1, 3, -2.0, "S"]]


FU1 = [[10, 0, 1.0], [11, 0, 0.2], [12, 0, 3.0]]
FU2 = [[20, 0, 0.5], [21, 0, 0.2], [22, 0, 0.45], [30, 0, 0.9]]
BASE1 = {"10", "11"}
BASE2 = {"20", "21", "22"}
RANK1 = {"10": 1, "11": 2}
RANK2 = {"20": 1, "21": 2, "22": 3}
TOP1 = {"10", "12"}
TOP2 = {"20", "21", "22"}


def _fixture():
    split = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": []},
             {"id": 2, "relevant_docs": [20, 21, 22], "relevant_chunks": []}]
    rr = {"1": _rows1(), "2": _rows2()}
    fu = {"1": FU1, "2": FU2}
    ref = {"1": [{"doc_id": 10, "chunk_text": "A"}, {"doc_id": 11, "chunk_text": "B"}],
           "2": [{"doc_id": 20, "chunk_text": "P"}, {"doc_id": 21, "chunk_text": "Q"},
                 {"doc_id": 20, "chunk_text": "S"}]}
    rank = {"1": RANK1, "2": RANK2}
    top100 = {"1": TOP1, "2": TOP2}
    return split, rr, fu, ref, rank, top100


def test_select_q_keeps_diag14_convention_drops_stolen_text():
    # "T" có dòng doc 12 (ngoài nền) s cao hơn -> order_q giữ dòng đó, lọc nền thì "T" biến mất
    got = select_q(_rows1(), FU1, BASE1, 3)
    assert got == [{"doc_id": 10, "chunk_text": "A"}, {"doc_id": 10, "chunk_text": "C"},
                   {"doc_id": 11, "chunk_text": "B"}]
    assert select_q(_rows1(), FU1, BASE1, 2) == got[:2]
    assert select_q(_rows1(), FU1, BASE1, 9) == got


def test_select_q_matches_diag14_c120in_pipeline():
    for rows, fu, base in ((_rows1(), FU1, BASE1), (_rows2(), FU2, BASE2)):
        split = [{"id": 1, "relevant_docs": [int(d) for d in sorted(base)], "relevant_chunks": []}]
        orders = {"1": order_q(rows, fu, 4.0)}
        out, _st = build_sub(split, orders, {}, "c120in", delta=30, cap=120)
        assert out[0]["relevant_chunks"] == select_q(rows, fu, base, len(out[0]["relevant_chunks"]))
    assert pick_q(_rows2(), FU2, 3, 4.0) == select_q(_rows2(), FU2, BASE2, 3)


def test_select_q_literal_keeps_text_under_base_doc():
    got = select_q_literal(_rows1(), FU1, BASE1, 3)
    assert got == [{"doc_id": 10, "chunk_text": "A"}, {"doc_id": 10, "chunk_text": "C"},
                   {"doc_id": 11, "chunk_text": "T"}]
    assert got != select_q(_rows1(), FU1, BASE1, 3)


def test_dup_text_stats_counts_rule_disagreement():
    order = order_q(_rows1(), FU1, 4.0)
    st = dup_text_stats(_rows1(), order)
    assert st == {"text_nhieu_dong": 1, "text_khac_doc": 1, "lech_luat_giu_dong": 1}
    assert dup_text_stats(_rows2(), order_q(_rows2(), FU2, 4.0)) == {
        "text_nhieu_dong": 0, "text_khac_doc": 0, "lech_luat_giu_dong": 0}


def test_filter_base_and_pair_set():
    order = [{"doc_id": 10, "chunk_text": "A", "s": 1.0}, {"doc_id": 12, "chunk_text": "X", "s": 0.5}]
    assert filter_base(order, BASE1) == order[:1]
    assert pair_set(order[:1]) == {("10", "A")}


def test_apply_bonus_first_chunk_of_each_doc():
    order = [{"doc_id": 10, "chunk_text": "A", "s": 5.0},
             {"doc_id": 10, "chunk_text": "C", "s": 4.6},
             {"doc_id": 11, "chunk_text": "B", "s": 4.0}]
    assert apply_bonus(order, 0.0) == order
    assert [c["chunk_text"] for c in apply_bonus(order, 1.0)] == ["A", "B", "C"]
    assert [c["chunk_text"] for c in apply_bonus(order, float("inf"))] == ["A", "B", "C"]
    assert [c["chunk_text"] for c in apply_bonus(order, 0.05)] == ["A", "C", "B"]


def test_select_q_with_bonus_changes_selection():
    sel0 = select_q(_rows1(), FU1, BASE1, 2, b=0.0)
    sel1 = select_q(_rows1(), FU1, BASE1, 2, b=1.0)
    assert [c["chunk_text"] for c in sel0] == ["A", "C"]
    assert [c["chunk_text"] for c in sel1] == ["A", "B"]
    sel_inf = select_q(_rows1(), FU1, BASE1, 3, b=float("inf"))
    assert [c["chunk_text"] for c in sel_inf] == ["A", "B", "C"]


def test_stats_q_bands_new_and_diff():
    sel = [{"doc_id": 5, "chunk_text": "a"}, {"doc_id": 35, "chunk_text": "b"},
           {"doc_id": 75, "chunk_text": "c"}, {"doc_id": 120, "chunk_text": "d"}]
    want = [{"doc_id": 5, "chunk_text": "a"}, {"doc_id": 5, "chunk_text": "e"}]
    rank = {"5": 5, "35": 35, "75": 75, "120": 120}
    st = stats_q(sel, want, rank, {"5", "35"}, k=6)
    assert st["n_chunk"] == 4 and st["n_doc"] == 4 and st["n_moi"] == 2
    assert st["n_khac"] == 3 and st["n_mat"] == 1 and st["n_thieu_k"] == 2
    assert st["bands"] == {"1-30": 1, "31-60": 1, "61-100": 1, "101+": 1}
    ag = aggregate([st])
    assert ag["TB_chunk_cau"] == 4.0 and ag["min_chunk"] == 4 and ag["max_chunk"] == 4
    assert ag["TB_doc_khac_nhau_co_chunk_cau"] == 4.0 and ag["TB_chunk_moi_cau"] == 2.0
    assert ag["TB_chunk_khac_c120in_cau"] == 3.0 and ag["TB_thieu_c120in_cau"] == 1.0
    assert ag["TB_lech_c120in_cau"] == 4.0 and ag["cau_khac_c120in"] == 1
    assert ag["cau_thieu_k_q"] == 1
    assert ag["phan_bang_hang_doc"]["101+"] == {"so_chunk": 1, "TB_moi_cau": 1.0}


def test_build_subs_two_queries_stats_and_bsweep():
    split, rr, fu, ref, rank, top100 = _fixture()
    sub_in, sub_cov, st = build_subs(split, rr, fu, ref, rank, top100, 4.0, 1.0)
    assert [c["chunk_text"] for c in sub_in[0]["relevant_chunks"]] == ["A", "C"]
    assert [c["chunk_text"] for c in sub_cov[0]["relevant_chunks"]] == ["A", "B"]
    assert [c["chunk_text"] for c in sub_in[1]["relevant_chunks"]] == ["P", "R", "S"]
    assert sub_in[0]["relevant_docs"] == [10, 11] and sub_cov[1]["relevant_docs"] == [20, 21, 22]
    for name, nd, nm, nk, nlech, nkhac in (("cx93in", 1.5, 0.0, 1.0, 2.0, 2), ("cx93cov", 2.0, 0.5, 0.5, 1.0, 1)):
        b = st["bai"][name]
        assert abs(b["TB_chunk_cau"] - 2.5) < 1e-9 and b["min_chunk"] == 2 and b["max_chunk"] == 3
        assert abs(b["TB_doc_khac_nhau_co_chunk_cau"] - nd) < 1e-9
        assert abs(b["TB_chunk_moi_cau"] - nm) < 1e-9
        assert abs(b["TB_chunk_khac_c120in_cau"] - nk) < 1e-9
        assert abs(b["TB_thieu_c120in_cau"] - nk) < 1e-9
        assert abs(b["TB_lech_c120in_cau"] - nlech) < 1e-9
        assert b["cau_khac_c120in"] == nkhac and b["cau_thieu_k_q"] == 0
        assert abs(b["TB_doc_cau"] - 2.5) < 1e-9 and b["TB_doc_them_cau"] == 0.0
        assert b["phan_bang_hang_doc"]["1-30"]["so_chunk"] == 5
    assert abs(st["k_q_TB"] - 2.5) < 1e-9
    assert st["trung_nguyen_van"] == {"text_nhieu_dong": 1, "text_khac_doc": 1, "lech_luat_giu_dong": 1}
    assert st["b_quet_TB_doc_co_chunk_cau"] == {"0.0": 1.5, "0.5": 1.5, "1.0": 2.0, "2.0": 2.0,
                                                "inf": 2.5}


def test_repro_check_ok_and_mismatch():
    split, rr, fu, _ref, _rank, _top = _fixture()
    base = {"1": BASE1, "2": BASE2}
    want = {"1": select_q(_rows1(), FU1, BASE1, 3), "2": select_q(_rows2(), FU2, BASE2, 4)}
    rep = repro_check(rr, fu, base, want, 4.0)
    assert rep["ok"] and rep["ty_le_cau_khop"] == 1.0 and rep["ty_le_vi_tri_chunk_khop"] == 1.0
    assert rep["bien_the_C_q_literal"]["ty_le_cau_khop"] == 0.5
    assert rep["trung_nguyen_van"] == {"text_nhieu_dong": 1, "text_khac_doc": 1, "lech_luat_giu_dong": 1}
    assert rep["cau_C_q_khac_k_q"] == 0
    bad = {"1": want["1"][:2] + [{"doc_id": 9, "chunk_text": "X"}], "2": want["2"]}
    rep2 = repro_check(rr, fu, base, bad, 4.0)
    assert not rep2["ok"] and rep2["ty_le_cau_khop"] == 0.5
    assert rep2["vi_du_lech"][0]["id"] == "1"
    rep3 = repro_check(rr, fu, base, {"1": want["1"] + [{"doc_id": 10, "chunk_text": "Z"}],
                                      "2": want["2"]}, 4.0)
    assert rep3["cau_C_q_khac_k_q"] == 1


def test_run_end_to_end_tiny(tmp_path):
    split, rr, fu, ref, _rank, _top = _fixture()
    pool = {"1": [[10, 0, 1.0], [12, 0, 0.5]], "2": [[20, 0, 1.0], [21, 0, 0.5], [22, 0, 0.2]]}
    files = {"rerank.json": rr, "fused.json": fu, "split.json": split, "ref.json": [
        {"id": q, "relevant_docs": [], "relevant_chunks": ch} for q, ch in ref.items()], "pool.json": pool}
    for name, obj in files.items():
        (tmp_path / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    rep = run(str(tmp_path / "rerank.json"), str(tmp_path / "split.json"), str(tmp_path / "fused.json"),
              str(tmp_path / "ref.json"), str(tmp_path / "pool.json"), str(out))
    assert rep["bai"]["cx93in"]["TB_chunk_cau"] == 2.5
    assert rep["bai"]["cx93cov"]["TB_chunk_moi_cau"] == 0.5
    assert rep["b_quet"]["inf"] == 2.5
    assert (out / "report.json").exists()
    for name in ("cx93in", "cx93cov"):
        sub = json.loads((out / f"submission_{name}.json").read_text(encoding="utf-8"))
        assert [it["id"] for it in sub] == [1, 2]
        assert all(str(c["doc_id"]) in {str(d) for d in it["relevant_docs"]}
                   for it in sub for c in it["relevant_chunks"])
        assert rep["bai"][name]["json_bytes"] == (out / f"submission_{name}.json").stat().st_size


def test_verify_zip_full_ok_and_limit(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": 1, "relevant_docs": [1], "relevant_chunks": []}]), encoding="utf-8")
    z = tmp_path / "s.zip"
    pack(p, z, method="lzma")
    st = verify_zip_full(str(p), str(z))
    assert st["ok"] and st["duoi_gioi_han"] and st["bang_goc_bytes"] and st["bang_goc_json"]
    assert st["zip_bytes"] == z.stat().st_size
    p2 = tmp_path / "s2.json"
    p2.write_text(json.dumps([{"id": 1, "relevant_docs": [2], "relevant_chunks": []}]), encoding="utf-8")
    st2 = verify_zip_full(str(p2), str(z))
    assert not st2["ok"]
