import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunk_docfuse import pick_q  # noqa: E402
from diag14 import analyze_bands, build_sub, order_q, repro_check, run, strip_meta, verify_zip  # noqa: E402
from zip_submission import pack  # noqa: E402


def _tiny():
    rows = [[10, 1, 0, 1.0, "A"], [11, 2, 1, 0.9, "B"], [12, 3, 2, 0.8, "C"], [10, 1, 3, 0.5, "A"]]
    fused = [[10, 0, 1.0], [11, 0, 0.5], [12, 0, 0.1]]
    return rows, fused


def test_order_q_matches_pick_q_prefix_and_dedup():
    rows, fused = _tiny()
    order = order_q(rows, fused, 4.0)
    assert [c["chunk_text"] for c in order] == ["A", "B", "C"]
    for k in (1, 2, 3):
        assert pick_q(rows, fused, k, 4.0) == strip_meta(order, k)
    assert pick_q(rows, fused, 9, 4.0) == strip_meta(order)


def test_order_q_meta_s_is_zc_plus_wd_zd():
    rows, fused = _tiny()
    order = order_q(rows, fused, 4.0)
    by = {c["chunk_text"]: c for c in order}
    assert abs(by["A"]["logit"] - 1.0) < 1e-9 and abs(by["C"]["logit"] - 0.8) < 1e-9
    for c in order:
        assert abs(c["s"] - (c["zc"] + 4.0 * c["zd"])) < 1e-9
    assert by["A"]["zd"] > by["B"]["zd"] > by["C"]["zd"]


def test_repro_check_exact_and_mismatch():
    rows, fused = _tiny()
    rr, fu = {"1": rows}, {"1": fused}
    ref = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{}, {}, {}]}]
    want = pick_q(rows, fused, 3, 4.0)
    target = [{"id": 1, "relevant_docs": [], "relevant_chunks": want}]
    rep = repro_check(rr, fu, ref, target, 4.0)
    assert rep["ok"] and rep["ty_le_cau_khop"] == 1.0 and rep["ty_le_vi_tri_chunk_khop"] == 1.0
    bad = [{"id": 1, "relevant_docs": [], "relevant_chunks": want[:2] + [{"doc_id": 9, "chunk_text": "X"}]}]
    rep2 = repro_check(rr, fu, ref, bad, 4.0)
    assert not rep2["ok"] and rep2["ty_le_cau_khop"] == 0.0
    assert 0.0 <= rep2["ty_le_vi_tri_chunk_khop"] < 1.0


def test_build_sub_modes_docs_and_filter():
    split = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": []}]
    orders = {"1": [{"doc_id": 10, "chunk_text": "A", "logit": 1.0, "zc": 0.0, "zd": 0.0, "s": 1.0},
                    {"doc_id": 11, "chunk_text": "B", "logit": 0.9, "zc": 0.0, "zd": 0.0, "s": 0.9},
                    {"doc_id": 12, "chunk_text": "C", "logit": 0.8, "zc": 0.0, "zd": 0.0, "s": 0.8},
                    {"doc_id": 13, "chunk_text": "D", "logit": 0.7, "zc": 0.0, "zd": 0.0, "s": 0.7}]}
    kq = {"1": 2}
    out, st = build_sub(split, orders, kq, "c120", delta=30, cap=120)
    assert [c["chunk_text"] for c in out[0]["relevant_chunks"]] == ["A", "B", "C", "D"]
    assert out[0]["relevant_docs"] == [10, 11, 12, 13] and st["TB_doc_them_cau"] == 2.0
    assert st["TB_chunk_cau"] == 4.0 and st["TB_doc_khac_nhau_co_chunk_cau"] == 4.0
    out, st = build_sub(split, orders, kq, "c120in", delta=30, cap=120)
    assert [c["chunk_text"] for c in out[0]["relevant_chunks"]] == ["A", "B"]
    assert out[0]["relevant_docs"] == [10, 11] and st["TB_doc_them_cau"] == 0.0
    out, st = build_sub(split, orders, kq, "ck105", delta=1, cap=120)
    assert [c["chunk_text"] for c in out[0]["relevant_chunks"]] == ["A", "B", "C"]
    assert out[0]["relevant_docs"] == [10, 11, 12]
    out, st = build_sub(split, orders, kq, "c120", delta=30, cap=2)
    assert [c["chunk_text"] for c in out[0]["relevant_chunks"]] == ["A", "B"]
    assert out[0]["relevant_docs"] == [10, 11]


def test_build_sub_unknown_mode_raises():
    try:
        build_sub([{"id": 1, "relevant_docs": [], "relevant_chunks": []}], {}, {}, "zzz")
        assert False
    except ValueError:
        pass


def test_analyze_bands_logit_zd_and_doc_membership():
    order = [{"doc_id": 1000 + i, "chunk_text": f"t{i}", "logit": float(i), "zc": 0.0,
              "zd": 0.5, "s": float(i)} for i in range(120)]
    orders = {"1": order}
    split_docs = {"1": {str(1000 + i) for i in range(75, 90)}}
    rep = analyze_bands(orders, {"1": 75}, split_docs)
    b1, b2, b3 = rep["76-90"], rep["91-105"], rep["106-120"]
    assert b1["so_chunk"] == 15 and b2["so_chunk"] == 15 and b3["so_chunk"] == 15
    assert abs(b1["TB_logit"] - 82.0) < 1e-9 and abs(b2["TB_logit"] - 97.0) < 1e-9
    assert abs(b3["TB_logit"] - 112.0) < 1e-9 and abs(b1["TB_zd"] - 0.5) < 1e-9
    assert b1["ti_le_doc_trong_danh_sach_nop"] == 1.0 and b2["ti_le_doc_trong_danh_sach_nop"] == 0.0
    assert b1["ti_le_thuc_su_them_hon_k_q"] == 1.0
    assert rep["tong"]["chunk_them_hon_k_q_toan_bo"] == 45
    rep2 = analyze_bands(orders, {"1": 100}, split_docs)
    assert abs(rep2["91-105"]["ti_le_thuc_su_them_hon_k_q"] - 5 / 15) < 1e-9


def test_verify_zip_ok_and_fail(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": 1, "relevant_docs": [1], "relevant_chunks": []}]), encoding="utf-8")
    z = tmp_path / "s.zip"
    pack(p, z, method="lzma")
    st = verify_zip(p, z)
    assert st["ok"] and st["bang_goc_bytes"] and st["bang_goc_json"] and st["entry"] == ["submission.json"]
    p2 = tmp_path / "s2.json"
    p2.write_text(json.dumps([{"id": 1, "relevant_docs": [2], "relevant_chunks": []}]), encoding="utf-8")
    st2 = verify_zip(p2, z)
    assert not st2["ok"] and not st2["bang_goc_json"]


def test_run_end_to_end_tiny(tmp_path):
    rows, fused = _tiny()
    rr, fu = {"1": rows}, {"1": fused}
    k = 2
    want = pick_q(rows, fused, k, 4.0)
    files = {
        "rerank.json": rr,
        "fused.json": fu,
        "ref.json": [{"id": 1, "relevant_docs": [], "relevant_chunks": [{} for _ in range(k)]}],
        "target.json": [{"id": 1, "relevant_docs": [], "relevant_chunks": want}],
        "split.json": [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": []}],
    }
    for name, obj in files.items():
        (tmp_path / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    outdir = tmp_path / "out"
    rep, code = run(str(tmp_path / "split.json"), str(tmp_path / "rerank.json"), str(tmp_path / "fused.json"),
                    str(tmp_path / "ref.json"), str(tmp_path / "target.json"), str(outdir),
                    wd=4.0, delta=30, cap=120)
    assert code == 0 and rep["repro"]["ok"]
    c120 = json.loads((outdir / "submission_c120.json").read_text(encoding="utf-8"))
    c120in = json.loads((outdir / "submission_c120in.json").read_text(encoding="utf-8"))
    ck = json.loads((outdir / "submission_ck105.json").read_text(encoding="utf-8"))
    assert [c["chunk_text"] for c in c120[0]["relevant_chunks"]] == ["A", "B", "C"]
    assert c120[0]["relevant_docs"] == [10, 11, 12]
    assert [c["chunk_text"] for c in c120in[0]["relevant_chunks"]] == ["A", "B"]
    assert c120in[0]["relevant_docs"] == [10, 11]
    assert [c["chunk_text"] for c in ck[0]["relevant_chunks"]] == ["A", "B", "C"]
    assert rep["bai"]["c120"]["TB_doc_them_cau"] == 1.0 and rep["bai"]["c120in"]["TB_doc_them_cau"] == 0.0
    assert rep["phan_tich_hang_s"]["tong"]["chunk_them_hon_k_q_toan_bo"] == 1


def test_run_aborts_when_repro_mismatch(tmp_path):
    rows, fused = _tiny()
    for name, obj in {
        "rerank.json": {"1": rows},
        "fused.json": {"1": fused},
        "ref.json": [{"id": 1, "relevant_docs": [], "relevant_chunks": [{}, {}, {}]}],
        "target.json": [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 9, "chunk_text": "X"}]}],
        "split.json": [{"id": 1, "relevant_docs": [10], "relevant_chunks": []}],
    }.items():
        (tmp_path / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    rep, code = run(str(tmp_path / "split.json"), str(tmp_path / "rerank.json"), str(tmp_path / "fused.json"),
                    str(tmp_path / "ref.json"), str(tmp_path / "target.json"), str(tmp_path / "out"))
    assert code == 2 and not rep["repro"]["ok"]
    assert not (tmp_path / "out" / "submission_c120.json").exists()
