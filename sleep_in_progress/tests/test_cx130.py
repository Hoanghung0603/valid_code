import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import cx130  # noqa: E402
from cx110 import norm_pairs  # noqa: E402
from cx130 import (build_sub, check_repro23, check_zip, largest_delta_within, run,  # noqa: E402
                   uoc_zip)
from cx93 import apply_bonus, filter_base, select_q  # noqa: E402
from diag14 import order_q, strip_meta  # noqa: E402
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
    ref110 = {"1": strip_meta(prep["1"]["ob"], 3), "2": strip_meta(prep["2"]["ob"], 4)}
    base = {"1": BASE1, "2": BASE2}
    rank = {"1": RANK1, "2": RANK2}
    dq = {"1": {"11"}, "2": {"21"}}
    return split, prep, ref110, dq, rank, base


def _rows28():
    return [[10, 1, i, float(28 - i), f"C{i:02d}"] for i in range(28)]


FU28 = [[10, 0, 1.0]]


def _fixture28():
    base = {"1": {"10"}}
    prep = {"1": _prep(_rows28(), FU28, {"10"}, 2)}
    return prep, base


def test_uoc_zip_theo_ti_le_cx110():
    assert uoc_zip(505_693_628) == pytest.approx(79_380_644)
    assert uoc_zip(0) == 0
    assert uoc_zip(505_693_628 * 2) == pytest.approx(79_380_644 * 2)
    assert uoc_zip(1000, json_ref=100, zip_ref=25) == 250


def test_largest_delta_within_budget():
    hist = [[10, 100, 20]]  # N(d) = 20 * min(10 + d, 100)
    # est(d) = 1000 * N(d) / 200 * 0.5 = 2.5 * N(d)
    assert largest_delta_within(hist, 1000, 200, 50, 1000, json_ref=100, zip_ref=50) == 10
    assert largest_delta_within(hist, 1000, 200, 50, 999, json_ref=100, zip_ref=50) == 9
    assert largest_delta_within(hist, 1000, 200, 5, 1000, json_ref=100, zip_ref=50) == 5
    assert largest_delta_within(hist, 1000, 200, 50, 0, json_ref=100, zip_ref=50) == 0
    assert largest_delta_within(hist, 1000, 200, 50, 5000, json_ref=100, zip_ref=50) == 50
    assert largest_delta_within(hist, 1000, 0, 50, 1000) == 0
    assert largest_delta_within(hist, 1000, 200, -1, 1000) == 0


def test_check_repro23_ok_and_mismatch():
    _split, prep, ref110, _dq, _rank, _base = _fixture()
    rep = check_repro23(prep, ref110)
    assert rep["ok"] and rep["ty_le_cau_khop"] == 1.0 and rep["ty_le_vi_tri_chunk_khop"] == 1.0
    assert rep["cau"] == 2 and rep["vi_du_lech"] == []
    bad = {"1": ref110["1"][:2] + [{"doc_id": 9, "chunk_text": "X"}], "2": ref110["2"]}
    rep2 = check_repro23(prep, bad)
    assert not rep2["ok"] and rep2["ty_le_cau_khop"] == 0.5
    assert rep2["vi_du_lech"][0]["id"] == "1"
    rep3 = check_repro23({}, ref110)
    assert not rep3["ok"] and rep3["cau"] == 2


def test_build_sub_saturated_tiny_prefix():
    split, prep, ref110, dq, rank, base = _fixture()
    sub, st = build_sub(split, prep, ref110, dq, rank, 23, base)
    assert [norm_pairs(it["relevant_chunks"]) for it in sub] == [norm_pairs(v) for v in ref110.values()]
    assert st["kprime"] == {"TB": 3.5, "min": 3, "max": 4, "tong": 7}
    assert st["chunk_them"]["so"] == 0 and st["TB_chunk_them_cau"] == 0.0
    assert st["cau_chan_boi_C_q"] == 2 and st["cau_tien_to_cx110cov_ok"] == 2
    sub2, st2 = build_sub(split, prep, ref110, dq, rank, 30, base)
    assert [norm_pairs(it["relevant_chunks"]) for it in sub2] == [norm_pairs(v) for v in ref110.values()]
    assert st2["cau_chan_boi_C_q"] == 2  # |C_q| = 3 < 2+30 va 4 < 3+30
    assert st2["kprime"] == st["kprime"] and st2["cau_tien_to_cx110cov_ok"] == 2
    assert st2["tien_to_kiem_tra"] and st2["TB_doc_cau"] == 2.5 and st2["TB_doc_them_cau"] == 0.0


def test_build_sub_extras_single_doc_exact():
    prep, base = _fixture28()
    ob = prep["1"]["ob"]
    assert [c["chunk_text"] for c in ob] == [f"C{i:02d}" for i in range(28)]
    ref110 = {"1": strip_meta(ob, 25)}
    split = [{"id": 1, "relevant_docs": [10], "relevant_chunks": []}]
    rank = {"1": {"10": 5}}
    dq = {"1": {"10"}}
    sub, st = build_sub(split, prep, ref110, dq, rank, 26, base)
    assert [c["chunk_text"] for c in sub[0]["relevant_chunks"]] == [f"C{i:02d}" for i in range(28)]
    assert norm_pairs(sub[0]["relevant_chunks"])[:25] == norm_pairs(ref110["1"])
    assert st["kprime"] == {"TB": 28, "min": 28, "max": 28, "tong": 28}
    assert st["cau_chan_boi_C_q"] == 0 and st["cau_tien_to_cx110cov_ok"] == 1
    assert st["TB_chunk_them_cau"] == 3.0 and st["chunk_them"]["so"] == 3
    assert st["chunk_them"]["phan_bang_hang_doc"]["1-30"] == {"so_chunk": 3, "TB_moi_cau": 3.0}
    assert st["chunk_them"]["phan_bang_hang_doc"]["31-60"]["so_chunk"] == 0
    assert st["chunk_them"]["TB_logit"] == pytest.approx(2.0)  # logit C25..C27 = 3, 2, 1
    exp_s = (ob[25]["s"] + ob[26]["s"] + ob[27]["s"]) / 3
    assert st["chunk_them"]["TB_s"] == pytest.approx(exp_s)
    assert st["chunk_them"]["nguon"] == {"moi_docwin_ext": 3, "cu": 0, "ti_le_moi": 1.0}
    assert st["TB_doc_khac_nhau_co_chunk_cau"] == 1.0
    _sub2, st2 = build_sub(split, prep, ref110, {"1": set()}, rank, 27, base)
    assert st2["chunk_them"]["nguon"] == {"moi_docwin_ext": 0, "cu": 3, "ti_le_moi": 0.0}
    assert st2["cau_chan_boi_C_q"] == 1  # 28 < 2+27


def test_build_sub_bands_nguon_two_docs():
    rows = [[10, 1, i, float(28 - i), f"C{i:02d}"] for i in range(26)] + \
           [[75, 2, 26, 2.0, "D0"], [75, 2, 27, 1.0, "D1"]]
    fu75 = [[10, 0, 1.0], [75, 0, 0.2]]
    base = {"1": {"10", "75"}}
    prep = {"1": _prep(rows, fu75, {"10", "75"}, 2)}
    ref110 = {"1": strip_meta(prep["1"]["ob"], 25)}
    split = [{"id": 1, "relevant_docs": [10, 75], "relevant_chunks": []}]
    rank = {"1": {"10": 5, "75": 75}}
    dq = {"1": {"75"}}
    sub, st = build_sub(split, prep, ref110, dq, rank, 26, base)
    extra = [c["chunk_text"] for c in sub[0]["relevant_chunks"][25:]]
    assert extra == ["C25", "D0", "D1"]
    assert st["chunk_them"]["so"] == 3 and st["TB_chunk_them_cau"] == 3.0
    assert st["chunk_them"]["phan_bang_hang_doc"]["1-30"] == {"so_chunk": 1, "TB_moi_cau": 1.0}
    assert st["chunk_them"]["phan_bang_hang_doc"]["61-100"] == {"so_chunk": 2, "TB_moi_cau": 2.0}
    assert st["chunk_them"]["TB_logit"] == pytest.approx(2.0)  # 3, 2, 1
    assert st["chunk_them"]["nguon"] == {"moi_docwin_ext": 2, "cu": 1, "ti_le_moi": pytest.approx(2 / 3)}
    assert st["TB_doc_khac_nhau_co_chunk_cau"] == 2.0


def _write_files(tmp_path, split, rr, fu, ref120, pool, dq, ref110):
    files = {"rerank.json": rr, "fused.json": fu, "split.json": split, "ref120.json": ref120,
             "pool.json": pool, "dq.json": dq, "ref110.json": ref110}
    for name, obj in files.items():
        (tmp_path / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return {k: str(tmp_path / k) for k in files}


def _files28(tmp_path, ref110_chunks):
    split = [{"id": 1, "relevant_docs": [10], "relevant_chunks": []}]
    ref120 = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"}] * 2}]
    ref110 = [{"id": 1, "relevant_docs": [10], "relevant_chunks": ref110_chunks}]
    return _write_files(tmp_path, split, {"1": _rows28()}, {"1": [[10, 0, 1.0]]}, ref120,
                        {"1": [[10, 0, 1.0]]}, {"1": [[11, 0, 0.0]]}, ref110)


def test_run_end_to_end_tiny(tmp_path):
    _prep28, _base = _fixture28()
    ref110_chunks = select_q(_rows28(), FU28, {"10"}, 25, b=1.0)
    p = _files28(tmp_path, ref110_chunks)
    out = tmp_path / "out"
    rep = run(p["rerank.json"], p["split.json"], p["fused.json"], p["ref120.json"], p["pool.json"],
              p["ref110.json"], p["dq.json"], str(out), target_mean=26.5)
    assert rep["tai_lap_delta23"]["ok"] and rep["tai_lap_delta23"]["ty_le_cau_khop"] == 1.0
    assert rep["delta"]["target"] == 25 and rep["delta"]["dung"] == 25 and rep["delta"]["dat_target"]
    assert not rep["delta"]["ghi_nhan"]
    assert "26.5" in rep["delta"]["ly_do"]
    assert rep["kprime"] == {"TB": 27.0, "min": 27, "max": 27, "tong": 27}
    assert rep["cau_chan_boi_C_q"] == 0
    st = rep["cx130cov"]
    assert st["cau_tien_to_cx110cov_ok"] == 1 and st["tien_to_kiem_tra"]
    assert st["TB_chunk_them_cau"] == 2.0 and st["chunk_them"]["so"] == 2
    assert st["chunk_them"]["TB_logit"] == pytest.approx(2.5)  # logit C25, C26 = 3, 2
    assert st["chunk_them"]["nguon"]["ti_le_moi"] == 0.0
    assert rep["hist_kcap"] == [[2, 28, 1]] and rep["k_q_TB"] == 2.0
    assert rep["uoc_zip"]["duoi_gioi_han"] and rep["uoc_zip"]["cac_lan"][0]["delta"] == 25
    sub = json.loads((out / "submission_cx130cov.json").read_text(encoding="utf-8"))
    assert [c["chunk_text"] for c in sub[0]["relevant_chunks"]] == [f"C{i:02d}" for i in range(27)]
    assert norm_pairs(sub[0]["relevant_chunks"])[:25] == norm_pairs(ref110_chunks)
    assert rep["cx130cov"]["json_bytes"] == (out / "submission_cx130cov.json").stat().st_size
    assert (out / "report.json").exists()

    rep23 = run(p["rerank.json"], p["split.json"], p["fused.json"], p["ref120.json"], p["pool.json"],
                p["ref110.json"], p["dq.json"], str(out), target_mean=26.5, delta=23)
    assert rep23["delta"]["dung"] == 23 and rep23["delta"]["ghi_nhan"]
    assert rep23["cx130cov"]["chunk_them"]["so"] == 0 and rep23["kprime"]["tong"] == 25
    sub23 = json.loads((out / "submission_cx130cov.json").read_text(encoding="utf-8"))
    assert norm_pairs(sub23[0]["relevant_chunks"]) == norm_pairs(ref110_chunks)


def test_run_budget_fallback_delta0(tmp_path):
    _prep28, _base = _fixture28()
    ref110_chunks = select_q(_rows28(), FU28, {"10"}, 25, b=1.0)
    p = _files28(tmp_path, ref110_chunks)
    out = tmp_path / "out"
    rep = run(p["rerank.json"], p["split.json"], p["fused.json"], p["ref120.json"], p["pool.json"],
              p["ref110.json"], p["dq.json"], str(out), target_mean=26.5, max_zip=1)
    assert rep["delta"]["target"] == 25 and rep["delta"]["dat_target"]
    assert rep["delta"]["dung"] == 0
    assert "uoc zip" in rep["delta"]["ly_do"] and "1" in rep["delta"]["ly_do"]
    assert not rep["uoc_zip"]["duoi_gioi_han"]
    assert not (out / "next_delta.txt").exists()  # delta da 0, khong giam duoc
    st = rep["cx130cov"]
    assert st["kprime"]["tong"] == 2 and not st["tien_to_kiem_tra"]
    assert st["chunk_them"]["so"] == 0
    sub = json.loads((out / "submission_cx130cov.json").read_text(encoding="utf-8"))
    assert len(sub[0]["relevant_chunks"]) == 2


def test_run_repro_fail_exits_2(tmp_path):
    _prep28, _base = _fixture28()
    ref110_chunks = select_q(_rows28(), FU28, {"10"}, 25, b=1.0)
    bad = [{"doc_id": 9, "chunk_text": "X"}] + ref110_chunks[1:]
    p = _files28(tmp_path, bad)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as e:
        run(p["rerank.json"], p["split.json"], p["fused.json"], p["ref120.json"], p["pool.json"],
            p["ref110.json"], p["dq.json"], str(out), target_mean=26.5)
    assert e.value.code == 2
    assert not (out / "submission_cx130cov.json").exists()
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert not rep["tai_lap_delta23"]["ok"]


def test_check_zip_ok_over_and_mismatch(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": 1, "relevant_docs": [1], "relevant_chunks": []}]), encoding="utf-8")
    z = tmp_path / "s.zip"
    pack(p, z, method="lzma")
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps({"delta": {"dung": 1, "thu": []}, "cx130cov": {"tong_chunk": 4},
                              "hist_kcap": [[2, 3, 1], [3, 4, 1]]}), encoding="utf-8")
    assert check_zip(str(p), str(z), str(rp), max_zip=10 ** 9) == 0
    rep = json.loads(rp.read_text(encoding="utf-8"))
    assert rep["zip"]["ok"] and rep["zip"]["duoi_gioi_han"] and rep["zip"]["bang_goc_bytes"]
    assert len(rep["zip"]["sha256_rut_gon"]) == 16 and len(rep["zip"]["sha256"]) == 64
    assert rep["delta"]["thu"][-1]["zip_bytes"] == z.stat().st_size

    assert check_zip(str(p), str(z), str(rp), max_zip=1) == 3
    assert (tmp_path / "next_delta.txt").read_text(encoding="utf-8") == "--delta 0"
    rep = json.loads(rp.read_text(encoding="utf-8"))
    assert not rep["zip"]["ok"] and len(rep["delta"]["thu"]) == 2

    rp.write_text(json.dumps({"delta": {"dung": 0, "thu": []}, "cx130cov": {"tong_chunk": 4},
                              "hist_kcap": [[2, 3, 1], [3, 4, 1]]}), encoding="utf-8")
    assert check_zip(str(p), str(z), str(rp), max_zip=1) == 4

    p.write_text(json.dumps([{"id": 1, "relevant_docs": [2], "relevant_chunks": []}]), encoding="utf-8")
    assert check_zip(str(p), str(z), str(rp), max_zip=10 ** 9) == 1
