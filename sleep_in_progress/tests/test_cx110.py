import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import cx110  # noqa: E402
from cx110 import (build_sub, check_repro, check_zip, choose_delta, hist_from_pairs, kprime,  # noqa: E402
                   norm_pairs, suggest_next_delta, total_chunks)
from cx93 import apply_bonus, filter_base, select_q  # noqa: E402
from diag14 import order_q  # noqa: E402
from zip_submission import pack  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cx93 import BASE1, BASE2, FU1, FU2, RANK1, RANK2, _rows1, _rows2  # noqa: E402


def _prep(rows, fu, base, k):
    cin = filter_base(order_q(rows, fu, 4.0), base)
    return {"kq": k, "cap": len(cin), "ob": apply_bonus(cin, 1.0)}


def _fixture():
    split = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": []},
             {"id": 2, "relevant_docs": [20, 21, 22], "relevant_chunks": []}]
    prep = {"1": _prep(_rows1(), FU1, BASE1, 2), "2": _prep(_rows2(), FU2, BASE2, 3)}
    ref93 = {"1": select_q(_rows1(), FU1, BASE1, 2, b=1.0),
             "2": select_q(_rows2(), FU2, BASE2, 3, b=1.0)}
    base = {"1": BASE1, "2": BASE2}
    rank = {"1": RANK1, "2": RANK2}
    dq = {"1": {"11"}, "2": {"21"}}
    return split, prep, ref93, dq, rank, base


def test_choose_delta_smallest_and_unreachable():
    pairs = [(2, 3), (3, 4)]
    assert choose_delta(pairs, 2.5) == (0, True)
    assert choose_delta(pairs, 3.5) == (1, True)
    assert choose_delta(pairs, 3.4) == (1, True)
    assert choose_delta(pairs, 3.6) == (1, False)  # TB |C_q| = 3,5 < 3,6
    assert choose_delta([(5, 5), (7, 9)], 6.0) == (0, True)
    assert choose_delta([(5, 5), (7, 9)], 6.5) == (1, True)
    assert choose_delta([(5, 5), (7, 9)], 8.0) == (2, False)
    assert choose_delta([(2, 3)], 0.0) == (0, True)


def test_kprime_hist_and_total_chunks():
    assert kprime(92, 130, 18) == 110
    assert kprime(92, 100, 18) == 100
    assert kprime(92, 130, 0) == 92
    hist = hist_from_pairs([(2, 3), (3, 4), (2, 3)])
    assert hist == [[2, 3, 2], [3, 4, 1]]
    assert total_chunks(hist, 0) == 2 * 2 + 3
    assert total_chunks(hist, 1) == 2 * 3 + 4
    assert total_chunks(hist, 5) == 2 * 3 + 4  # chan boi |C_q|


def test_build_sub_delta0_prefix_and_extras():
    split, prep, ref93, dq, rank, base = _fixture()
    sub0, st0 = build_sub(split, prep, ref93, dq, rank, 0, base)
    assert norm_pairs(sub0[0]["relevant_chunks"]) == norm_pairs(ref93["1"])
    assert norm_pairs(sub0[1]["relevant_chunks"]) == norm_pairs(ref93["2"])
    assert st0["kprime"] == {"TB": 2.5, "min": 2, "max": 3, "tong": 5}
    assert st0["chunk_them"]["so"] == 0 and st0["cau_chan_boi_C_q"] == 0

    sub1, st1 = build_sub(split, prep, ref93, dq, rank, 1, base)
    assert st1["kprime"] == {"TB": 3.5, "min": 3, "max": 4, "tong": 7}
    assert st1["cau_chan_boi_C_q"] == 0 and st1["cau_tien_to_cx93cov_ok"] == 2
    for it, ref, k in ((sub1[0], ref93["1"], 2), (sub1[1], ref93["2"], 3)):
        assert norm_pairs(it["relevant_chunks"])[:k] == norm_pairs(ref)
    assert [c["chunk_text"] for c in sub1[0]["relevant_chunks"]] == ["A", "B", "C"]
    assert [c["chunk_text"] for c in sub1[1]["relevant_chunks"]] == ["P", "R", "S", "Q"]
    assert st1["TB_chunk_them_cau"] == 1.0 and st1["chunk_them"]["so"] == 2
    assert st1["chunk_them"]["phan_bang_hang_doc"]["1-30"] == {"so_chunk": 2, "TB_moi_cau": 1.0}
    assert st1["chunk_them"]["phan_bang_hang_doc"]["31-60"] == {"so_chunk": 0, "TB_moi_cau": 0.0}
    assert st1["chunk_them"]["TB_logit"] == pytest.approx(0.7)
    exp_s = (prep["1"]["ob"][2]["s"] + prep["2"]["ob"][3]["s"]) / 2
    assert st1["chunk_them"]["TB_s"] == pytest.approx(exp_s)
    assert st1["chunk_them"]["nguon"] == {"moi_docwin_ext": 1, "cu": 1, "ti_le_moi": 0.5}
    assert st1["TB_doc_khac_nhau_co_chunk_cau"] == 2.5
    assert st1["TB_doc_cau"] == 2.5 and st1["TB_doc_them_cau"] == 0.0

    sub2, st2 = build_sub(split, prep, ref93, dq, rank, 2, base)
    assert st2["cau_chan_boi_C_q"] == 2  # |C_q| = 3 < 2+2 va 4 < 3+2
    assert st2["kprime"] == st1["kprime"]


def test_check_repro_ok_and_mismatch():
    _split, prep, ref93, _dq, _rank, _base = _fixture()
    rep = check_repro(prep, ref93)
    assert rep["ok"] and rep["ty_le_cau_khop"] == 1.0 and rep["ty_le_vi_tri_chunk_khop"] == 1.0
    assert rep["cau"] == 2 and rep["vi_du_lech"] == []
    bad = {"1": ref93["1"][:1] + [{"doc_id": 9, "chunk_text": "X"}], "2": ref93["2"]}
    rep2 = check_repro(prep, bad)
    assert not rep2["ok"] and rep2["ty_le_cau_khop"] == 0.5
    assert rep2["vi_du_lech"][0]["id"] == "1"
    rep3 = check_repro({}, ref93)
    assert not rep3["ok"] and rep3["cau"] == 2


def _write_inputs(tmp_path, ref93, k1=2, k2=3):
    split, rr, fu = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": []},
                     {"id": 2, "relevant_docs": [20, 21, 22], "relevant_chunks": []}], \
        {"1": _rows1(), "2": _rows2()}, {"1": FU1, "2": FU2}
    ref120 = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"}] * k1},
              {"id": 2, "relevant_docs": [], "relevant_chunks": [{"doc_id": 20, "chunk_text": "y"}] * k2}]
    pool = {"1": [[10, 0, 1.0], [12, 0, 0.5]], "2": [[20, 0, 1.0], [21, 0, 0.5], [22, 0, 0.2]]}
    dq = {"1": [[11, 0, 0.0]], "2": [[21, 0, 0.0]]}
    files = {"rerank.json": rr, "fused.json": fu, "split.json": split, "ref120.json": ref120,
             "pool.json": pool, "dq.json": dq,
             "ref93.json": [{"id": int(q), "relevant_docs": [], "relevant_chunks": ch}
                            for q, ch in ref93.items()]}
    for name, obj in files.items():
        (tmp_path / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return {k: str(tmp_path / k) for k in files}


def test_run_end_to_end_tiny(tmp_path):
    ref93 = {"1": select_q(_rows1(), FU1, BASE1, 2, b=1.0),
             "2": select_q(_rows2(), FU2, BASE2, 3, b=1.0)}
    p = _write_inputs(tmp_path, ref93)
    out = tmp_path / "out"
    rep = cx110.run(p["rerank.json"], p["split.json"], p["fused.json"], p["ref120.json"], p["pool.json"],
                    p["ref93.json"], p["dq.json"], str(out), target_mean=3.5)
    assert rep["tai_lap_delta0"]["ok"] and rep["tai_lap_delta0"]["ty_le_cau_khop"] == 1.0
    assert rep["delta"]["target"] == 1 and rep["delta"]["dung"] == 1 and rep["delta"]["dat_target"]
    assert rep["kprime"] == {"TB": 3.5, "min": 3, "max": 4, "tong": 7}
    assert rep["cau_chan_boi_C_q"] == 0
    assert rep["cx110cov"]["cau_tien_to_cx93cov_ok"] == 2
    assert rep["cx110cov"]["chunk_them"]["nguon"]["ti_le_moi"] == 0.5
    assert rep["hist_kcap"] == [[2, 3, 1],  [3, 4, 1]]
    assert rep["k_q_TB"] == 2.5
    sub = json.loads((out / "submission_cx110cov.json").read_text(encoding="utf-8"))
    assert [it["id"] for it in sub] == [1, 2]
    for it, k in ((sub[0], 2), (sub[1], 3)):
        assert norm_pairs(it["relevant_chunks"])[:k] == norm_pairs(ref93[str(it["id"])])
        assert all(str(c["doc_id"]) in {str(d) for d in it["relevant_docs"]} for c in it["relevant_chunks"])
    assert rep["cx110cov"]["json_bytes"] == (out / "submission_cx110cov.json").stat().st_size
    assert (out / "report.json").exists()

    rep0 = cx110.run(p["rerank.json"], p["split.json"], p["fused.json"], p["ref120.json"], p["pool.json"],
                     p["ref93.json"], p["dq.json"], str(out), target_mean=3.5, delta=0)
    assert rep0["delta"]["dung"] == 0 and rep0["delta"]["target"] == 1
    assert rep0["kprime"]["TB"] == 2.5 and rep0["cx110cov"]["chunk_them"]["so"] == 0
    assert rep0["delta"]["thu"] == []


def test_run_repro_fail_exits_2(tmp_path):
    ref93 = {"1": select_q(_rows1(), FU1, BASE1, 1, b=1.0)[:1], "2": select_q(_rows2(), FU2, BASE2, 3, b=1.0)}
    ref93["1"] = [{"doc_id": 9, "chunk_text": "X"}]
    p = _write_inputs(tmp_path, ref93)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as e:
        cx110.run(p["rerank.json"], p["split.json"], p["fused.json"], p["ref120.json"], p["pool.json"],
                  p["ref93.json"], p["dq.json"], str(out))
    assert e.value.code == 2
    assert not (out / "submission_cx110cov.json").exists()
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert not rep["tai_lap_delta0"]["ok"]


def test_suggest_next_delta_uses_chunk_budget():
    hist = [[10, 100, 20]]
    assert suggest_next_delta(hist, 5, 300, 3000, 2400) == 2
    assert suggest_next_delta(hist, 5, 300, 3000, 2400, marg=8.0) == 1
    assert suggest_next_delta(hist, 1, 240, 3000, 2900) == 0
    assert suggest_next_delta(hist, 0, 200, 3000, 1) == 0


def test_check_zip_ok_over_and_mismatch(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": 1, "relevant_docs": [1], "relevant_chunks": []}]), encoding="utf-8")
    z = tmp_path / "s.zip"
    pack(p, z, method="lzma")
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps({"delta": {"dung": 1, "thu": []}, "cx110cov": {"tong_chunk": 4},
                              "hist_kcap": [[2, 3, 1], [3, 4, 1]]}), encoding="utf-8")
    monkeypatch.setattr(cx110, "ZIP_LIMIT", 10 ** 9)
    assert check_zip(str(p), str(z), str(rp)) == 0
    rep = json.loads(rp.read_text(encoding="utf-8"))
    assert rep["zip"]["ok"] and rep["zip"]["duoi_gioi_han"] and rep["zip"]["bang_goc_bytes"]
    assert len(rep["zip"]["sha256_rut_gon"]) == 16 and len(rep["zip"]["sha256"]) == 64
    assert rep["delta"]["thu"][-1]["zip_bytes"] == z.stat().st_size

    monkeypatch.setattr(cx110, "ZIP_LIMIT", 1)
    assert check_zip(str(p), str(z), str(rp)) == 3
    assert (tmp_path / "next_delta.txt").read_text(encoding="utf-8") == "--delta 0"
    rep = json.loads(rp.read_text(encoding="utf-8"))
    assert not rep["zip"]["ok"] and len(rep["delta"]["thu"]) == 2

    rp.write_text(json.dumps({"delta": {"dung": 0, "thu": []}, "cx110cov": {"tong_chunk": 4},
                              "hist_kcap": [[2, 3, 1], [3, 4, 1]]}), encoding="utf-8")
    assert check_zip(str(p), str(z), str(rp)) == 4

    p.write_text(json.dumps([{"id": 1, "relevant_docs": [2], "relevant_chunks": []}]), encoding="utf-8")
    monkeypatch.setattr(cx110, "ZIP_LIMIT", 10 ** 9)
    assert check_zip(str(p), str(z), str(rp)) == 1
