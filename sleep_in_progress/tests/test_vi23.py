"""Test vi23.py - dua doc VI moi vao he truy hoi + dung bai #5532 (diag23)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import vi23
from vi23 import (
    build_pool_v23,
    chain_build,
    compare_subs,
    ext_coverage,
    filter_records,
    first_n_docs,
    fit_new_hosts,
    full8b_docs,
    hit_ids,
    make_body,
    merge_topk,
    new_pool_docs,
)


# ---------------------------------------------------------------- tien ich fixture

def test_check_zip_over_budget_uses_saved_counted_histogram(tmp_path):
    import zipfile

    source = tmp_path / "submission.json"
    source.write_text('[{"id": 1, "relevant_docs": [], "relevant_chunks": []}]', encoding="utf-8")
    packed = tmp_path / "submission.zip"
    with zipfile.ZipFile(packed, "w", zipfile.ZIP_LZMA) as archive:
        archive.write(source, "submission.json")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"delta": 57, "tong_chunk": 68,
                                  "hist_kcap": [[10, 100, 1], [20, 21, 1]]}), encoding="utf-8")
    assert vi23.check_zip(str(source), str(packed), str(report), max_zip=1) == 3
    result = json.loads(report.read_text(encoding="utf-8"))["zip"]["vi23"]
    assert result["next_delta"] < 57
    assert (tmp_path / "next_delta_vi23.txt").read_text() == f'--delta {result["next_delta"]}'

def rec(did, host="a.vn", text="x" * 300, status=200, title="t"):
    return {"id": did, "url": f"http://{host}/{did}", "final_url": f"http://{host}/{did}",
            "host": host, "status": status, "title": title, "text": text, "date": "", "ts": 0}


def hits(*rows):
    return [list(r) for r in rows]


# ---------------------------------------------------------------- filter_records

def test_filter_records_loc_theo_status_text_ngon_ngu_trung():
    rows = [
        rec(1, text="tieng viet du " + "a" * 300),
        rec(2, status=404),
        rec(3, text="ngan"),
        rec(4, text="这是中文内容" * 50),
        rec(1, text="trung id " + "b" * 300),
        rec(5, text="tieng viet " + "c" * 300),
    ]
    kept, st, ids = filter_records(rows)
    assert ids == [1, 5]
    assert [r["id"] for r in kept] == [1, 5]
    assert st["tong"] == 6 and st["giu"] == 2
    assert st["loai"] == {"khong_200": 1, "text_ngan": 1, "khong_phai_vi": 1, "trung_id": 1, "thieu_id": 0}
    assert st["theo_host"] == {"a.vn": 2}


def test_filter_records_seen_ids_ngoai_vao():
    kept, st, ids = filter_records([rec(7, text="v" * 300)], seen_ids={7})
    assert kept == [] and st["loai"]["trung_id"] == 1


def test_filter_records_thieu_id():
    kept, st, ids = filter_records([{"text": "v" * 300, "status": 200, "host": "a"}])
    assert kept == [] and st["loai"]["thieu_id"] == 1


# ---------------------------------------------------------------- boilerplate

def test_fit_new_hosts_bp_giu_host_cu_fit_host_moi():
    old = {"cu.vn": ["menu"], "khac.vn": ["footer"]}
    rows = ([rec(1, host="cu.vn", text=f"dong lap\nnoi dung {i} " + "v" * 50) for i in range(80)]
            + [rec(200, host="moi.vn", text=f"dong lap\nnoi dung {i} " + "v" * 50) for i in range(80)]
            + [rec(300, host="it.vn", text="noi dung " + "v" * 300)])
    merged, rep = fit_new_hosts(old, rows)
    assert merged["cu.vn"] == ["menu"]          # khong refit host cu
    assert merged["khac.vn"] == ["footer"]
    assert "moi.vn" in rep["host_moi_duoc_fit"]
    assert "it.vn" in rep["host_moi_it_mau_duoi_50"]
    assert rep["host_moi_so_doc"]["moi.vn"] == 80
    assert merged["moi.vn"] == ["dong lap"]      # dong lap > 20% -> boilerplate
    assert "it.vn" not in merged


# ---------------------------------------------------------------- dense merge

def test_merge_topk_giu_diem_cu_xep_theo_cosine():
    old = hits([1, 10, 0.9], [2, 11, 0.8], [3, 12, 0.5])
    new = hits([101, 50, 0.85], [102, 51, 0.95])
    out, st = merge_topk(old, new, top=3)
    assert [h[0] for h in out] == [102, 1, 101]
    assert out[0][2] == 0.95 and out[1][2] == 0.9      # diem cu giu nguyen
    assert st["so_cu"] == 1 and st["so_moi"] == 2


def test_merge_topk_trung_diem_hit_cu_len_truoc():
    old = hits([1, 10, 0.8])
    new = hits([9, 50, 0.8])
    out, _ = merge_topk(old, new, top=2)
    assert [h[0] for h in out] == [1, 9]


def test_merge_topk_khong_moi_giong_cu():
    old = hits([1, 10, 0.9], [2, 11, 0.8])
    out, st = merge_topk(old, [], top=200)
    assert out == [list(h) for h in old]
    assert st["so_moi"] == 0 and st["day_het"] is False


def test_merge_topk_score_none_xuong_cuoi():
    old = hits([1, 10, None], [2, 11, 0.8])
    out, _ = merge_topk(old, [], top=2)
    assert [h[0] for h in out] == [2, 1]


# ---------------------------------------------------------------- pool

def test_new_pool_docs_chi_lay_doc_moi():
    bm25 = hits([1, 1, 5.0], [200, 2, 4.0], [201, 3, 3.0])
    dense = hits([2, 11, 0.9], [202, 12, 0.8], [201, 13, 0.7])
    rows, st = new_pool_docs(bm25, dense, old_pool_ids={"1", "2"}, new_ids={"200", "201", "202"},
                             idx_of={"200": 500, "201": 501, "202": 502})
    assert [r[0] for r in rows] == [200, 201, 202]
    assert [r[1] for r in rows] == [500, 501, 502]
    assert all(r[2] is None for r in rows)
    assert st["bm25"] == 2 and st["dense"] == 1 and st["bm25+dense"] == 1


def test_new_pool_docs_idx_thay_the():
    rows, _ = new_pool_docs(hits([9, 7, 1.0]), [], old_pool_ids=set(), new_ids={"9"}, idx_of={})
    assert rows == [[9, 7, None]]


def test_build_pool_v23_giu_hit_cu_gop_logit_moi():
    old = {"1": hits([1, 10, 3.0], [2, 11, 1.0])}
    new = {"1": hits([300, 60, None], [301, 61, None]), "2": hits([400, 70, None])}
    logits = {"1": {"300": 2.5}, "2": {"400": 0.5}}
    out = build_pool_v23(old, new, logits)
    assert [h[0] for h in out["1"]] == [1, 300, 2, 301]     # 3.0 > 2.5 > 1.0 > None
    assert out["1"][1][2] == 2.5 and out["1"][0][2] == 3.0 and out["1"][2][2] == 1.0
    assert out["2"] == [[400, 70, 0.5]]


def test_build_pool_v23_logit_none_xuong_cuoi_va_bo_trung():
    old = {"1": hits([1, 10, 3.0])}
    new = {"1": hits([1, 99, None], [2, 11, None])}
    out = build_pool_v23(old, new, {})
    assert [h[0] for h in out["1"]] == [1, 2]
    assert out["1"][1] == [2, 11, None]


# ---------------------------------------------------------------- doclist

def test_hit_ids_va_first_n_docs():
    h = hits([3, 0, 1.0], [1, 0, 1.0], [3, 0, 1.0], [2, 0, 1.0])
    assert hit_ids(h) == [3, 1, 2]
    assert first_n_docs(h, 2) == [3, 1]


def test_full8b_docs_them_doc_nguon_chunk():
    base = [1, 2]
    ref = [{"doc_id": 2, "chunk_text": "a"}, {"doc_id": 9, "chunk_text": "b"}, {"doc_id": 9, "chunk_text": "c"}]
    assert full8b_docs(base, ref) == [1, 2, 9]


# ---------------------------------------------------------------- chain_build

def tiny_chain_fixture():
    """2 cau; pool 3 doc/cau; cands chunk thuoc cac doc do."""
    pool = {
        "1": hits([10, 1, 3.0], [11, 2, 1.0], [12, 3, -5.0]),
        "2": hits([20, 4, 2.0], [21, 5, 1.5], [22, 6, 1.0]),
    }
    q8b = {"1": hits([10, 1, 1.0]), "2": hits([20, 4, 1.0])}
    dense = {"1": hits([11, 2, 0.9], [10, 1, 0.8]), "2": hits([21, 5, 0.9], [20, 4, 0.8])}
    hyde = {"1": hits([12, 3, 0.7]), "2": hits([22, 6, 0.7])}
    cands = {
        "1": [[10, 1, 0, 5.0, "chunk a"], [11, 2, 0, 4.0, "chunk b"],
              [12, 3, 0, 3.0, "chunk c"], [10, 1, 1, 2.0, "chunk d"]],
        "2": [[20, 1, 0, 5.0, "chunk e"], [21, 2, 0, 1.0, "chunk f"]],
    }
    kq = {"1": 2, "2": 1}
    ref_chunks = {"1": [{"doc_id": 9, "chunk_text": "x"}], "2": []}
    return pool, q8b, dense, hyde, cands, kq, ref_chunks


def test_chain_build_doclist_va_chunk():
    pool, q8b, dense, hyde, cands, kq, ref = tiny_chain_fixture()
    rrf, fhyde, sub, rep = chain_build(pool, q8b, dense, hyde, cands, kq, ref,
                                       thr=-2.0, lo=1, hi=10, wd=4.0, b_cov=1.0, delta=2)
    # n_q = adaptive_n: cau 1 co 2 doc > -2.0 (3.0; 1.0) -> 2; cau 2: 3 doc -> 3
    assert rep["n_q_min"] == 2 and rep["n_q_max"] == 3
    s1, s2 = sub[0], sub[1]
    assert str(s1["id"]) == "1"
    # nen = 2 doc dau fhyde [10, 11]; them doc 9 (nguon chunk ref) -> [10, 11, 9]
    assert [str(d) for d in s1["relevant_docs"]] == ["10", "11", "9"]
    # C_q = 3 chunk (doc 10 x2, doc 11 x1); k' = min(2+2, 3) = 3
    assert len(s1["relevant_chunks"]) == 3
    assert {str(c["doc_id"]) for c in s1["relevant_chunks"]} == {"10", "11"}
    assert all(str(c["doc_id"]) in {"10", "11", "9"} for c in s1["relevant_chunks"])
    # cau 2: nen = 3 doc dau fhyde; C_q = 2 chunk, k' = min(1+2, 2) = 2
    assert [str(d) for d in s2["relevant_docs"]] == ["20", "21", "22"]
    assert len(s2["relevant_chunks"]) == 2
    assert rep["tong_chunk"] == 5
    # fusion tinh dung RRF k60: doc 10 = 1/61 + 1/61 + 2/62
    assert rrf["1"][0][0] == 10
    assert abs(rrf["1"][0][2] - (1 / 61 + 1 / 61 + 2 / 62)) < 1e-8
    assert abs(fhyde["1"][2][2] - (1 / 63 + 1 / 61)) < 1e-8   # doc 12: bge rank 3 + hyde rank 1


def test_chain_build_thuong_b_chon_chunk_dau_moi_doc():
    # doc 10: chunk a (logit 5.0), d (4.0); doc 11: chunk b (3.5) - wd=0 de doc lap theo logit
    cands = {"1": [[10, 1, 0, 5.0, "a"], [10, 1, 1, 4.0, "d"], [11, 2, 0, 3.5, "b"]]}
    pool = {"1": hits([10, 1, 3.0], [11, 2, 1.0])}
    _, _, sub, _ = chain_build(pool, {}, {}, {}, cands, {"1": 2}, {"1": []},
                               thr=-2.0, lo=1, hi=10, wd=0.0, b_cov=1.0, delta=0)
    # k' = min(2+0, 3) = 2; khong b: [a, d]; co b = 1: [a, b] (chunk dau moi doc duoc thuong)
    assert [c["chunk_text"] for c in sub[0]["relevant_chunks"]] == ["a", "b"]


def test_chain_build_xac_dinh_khong_lech():
    pool, q8b, dense, hyde, cands, kq, ref = tiny_chain_fixture()
    a = chain_build(pool, q8b, dense, hyde, cands, kq, ref, 2.0, 1, 10, 4.0, 1.0, 57)
    b = chain_build(pool, q8b, dense, hyde, cands, kq, ref, 2.0, 1, 10, 4.0, 1.0, 57)
    assert a[2] == b[2]


def test_compare_subs_phat_hien_lech():
    sub = [{"id": 1, "relevant_docs": [1, 2], "relevant_chunks": [{"doc_id": 1, "chunk_text": "a"}]}]
    ref = [{"id": 1, "relevant_docs": [1, 2], "relevant_chunks": [{"doc_id": 1, "chunk_text": "a"}]}]
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ref, f)
        p = f.name
    assert compare_subs(sub, p)["ok"] is True
    sub[0]["relevant_chunks"][0]["chunk_text"] = "b"
    assert compare_subs(sub, p)["ok"] is False


# ---------------------------------------------------------------- docwin / export / khac

def test_ext_coverage():
    pool = hits(*[[i, i, 1.0] for i in range(150)])
    assert ext_coverage(pool, [900, 901], top_n=100) == {str(i) for i in range(100)} | {"900", "901"}


def test_make_body():
    assert make_body("t", "noi dung", 100) == "t\nnoi dung"
    assert make_body("t", "noi dung", 100, no_title=True) == "noi dung"
    assert make_body("t", "abcdefghij", 5) == "t\nabc"


def test_cmplogit_local():
    got = {"1": [[10, 1, 2.0], [11, 2, 1.0]]}
    ref = {"1": {"10": 2.0, "11": 1.005}}
    rep = vi23.cmplogit_local(got, ref, tol=0.01)
    assert rep["ok"] is True and abs(rep["max_sai"] - 0.005) < 1e-9
    ref2 = {"1": {"10": 2.0, "11": 0.5}, "2": {"9": 1.0}}
    rep2 = vi23.cmplogit_local(got, ref2, tol=0.01)
    assert rep2["ok"] is False and rep2["so_cap_thieu_trong_got"] == 1


def test_bm25_diff():
    old = {"1": hits([1, 0, 9.0], [2, 0, 8.0], [3, 0, 7.0])}
    new = {"1": hits([2, 0, 9.0], [3, 0, 8.0], [300, 0, 7.0])}
    rep = vi23.bm25_diff(old, new, new_ids={"300"}, top=3)
    assert rep["TB_doc_cu_mat_khoi_top200_vi23"] == 1
    assert rep["TB_doc_cu_trung_top200"] == 2
    assert rep["TB_doc_moi_trong_top200_vi23"] == 1


# ---------------------------------------------------------------- cmd voi file tam

def test_cmd_prep_va_pool_pipeline(tmp_path):
    import zstandard
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rows = [rec(11, host="moi.vn", text="v" * 300), rec(12, host="moi.vn", text="v" * 300),
            rec(13, status=500, text="v" * 300), rec(11, host="moi.vn", text="v" * 300)]
    with (shard_dir / "vi-recov-00000.jsonl.zst").open("wb") as f:
        f.write(zstandard.ZstdCompressor().compress(
            ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)).encode("utf-8")))
    old_bp = tmp_path / "old_bp.json"
    old_bp.write_text(json.dumps({"cu.vn": ["m"]}), encoding="utf-8")
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"moi.vn": [11, 12, 13]}), encoding="utf-8")
    out = tmp_path / "out"
    rc = vi23.cmd_prep(type("A", (), {
        "shards": str(shard_dir), "pattern": "*.jsonl.zst", "old_bp": str(old_bp),
        "missing": str(missing), "old_ids": "", "out_dir": str(out),
        "bp_out": str(tmp_path / "vi23"), "report": ""})())
    assert rc == 0
    rep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert rep["giu"] == 2 and rep["loai"]["khong_200"] == 1
    assert json.loads((out / "new_ids.json").read_text(encoding="utf-8")) == [11, 12]
    assert json.loads((tmp_path / "vi23" / "boilerplate.json").read_text(encoding="utf-8"))["cu.vn"] == ["m"]

    # pool-new + pool-merge
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"1": [[1, 1, 3.0]]}), encoding="utf-8")
    bm25 = tmp_path / "bm25.json"
    bm25.write_text(json.dumps({"1": [[1, 1, 9.0], [11, 2, 8.0]]}), encoding="utf-8")
    dense = tmp_path / "dense.json"
    dense.write_text(json.dumps({"1": [[12, 3, 0.9], [1, 1, 0.8]]}), encoding="utf-8")
    args = type("A", (), {"pool": str(pool), "bm25_new": str(bm25), "dense_new": str(dense),
                          "new_ids": str(out / "new_ids.json"), "idx": "",
                          "out_cands": str(tmp_path / "cands.json"),
                          "out_new_hits": str(tmp_path / "new_hits.json"),
                          "report": str(tmp_path / "pn.json")})()
    assert vi23.cmd_pool_new(args) == 0
    cands = json.loads((tmp_path / "cands.json").read_text(encoding="utf-8"))
    assert cands["1"] == [[11, 2], [12, 3]]
    rr = tmp_path / "rr.json"
    rr.write_text(json.dumps({"1": [[11, 2, 1.25], [12, 3, -3.0]]}), encoding="utf-8")
    args2 = type("A", (), {"pool_old": str(pool), "new_hits": str(tmp_path / "new_hits.json"),
                           "rerank": str(rr), "out": str(tmp_path / "pool_v23.json"),
                           "report": str(tmp_path / "pm.json")})()
    assert vi23.cmd_pool_merge(args2) == 0
    merged = json.loads((tmp_path / "pool_v23.json").read_text(encoding="utf-8"))
    assert [h[0] for h in merged["1"]] == [1, 11, 12]        # 3.0 > 1.25 > -3.0
    assert merged["1"][0] == [1, 1, 3.0]
    pm = json.loads((tmp_path / "pm.json").read_text(encoding="utf-8"))
    assert pm["pool_cu_giu_nguyen"] is True


def test_cmd_docwin_cands_tach_p3_va_idx_map(tmp_path):
    doclist = tmp_path / "doclist.json"
    doclist.write_text(json.dumps({"1": [1, 5, 6, 7]}), encoding="utf-8")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"1": [[1, 1, 3.0]]}), encoding="utf-8")   # top-100 = {1}
    split = tmp_path / "split.json"
    split.write_text(json.dumps([{"id": 1, "relevant_docs": [2], "relevant_chunks": []}]), encoding="utf-8")
    p3w = tmp_path / "p3w.json"
    p3w.write_text(json.dumps({"1": {"5": [[0, 10, 0.5]]}}), encoding="utf-8")
    p3c = tmp_path / "p3c.json"
    p3c.write_text(json.dumps({"1": [[5, 50]]}), encoding="utf-8")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([6, 7]), encoding="utf-8")
    idxmap = tmp_path / "idx.json"
    idxmap.write_text(json.dumps({"5": 500, "6": 600, "7": 700}), encoding="utf-8")
    out = tmp_path / "out"
    args = type("A", (), {"doclist": str(doclist), "pool": str(pool), "split": str(split),
                          "p3_docwin": str(p3w), "p3_cands": str(p3c), "new_ids": str(ids),
                          "idx_from": [str(idxmap)], "top_n": 100, "out_dir": str(out)})()
    assert vi23.cmd_docwin_cands(args) == 0
    reuse = json.loads((out / "cands_docwin_p3reuse.json").read_text(encoding="utf-8"))
    new = json.loads((out / "cands_docwin_vi23.json").read_text(encoding="utf-8"))
    k_new = json.loads((out / "cache_docwin_new.json").read_text(encoding="utf-8"))
    k_old = json.loads((out / "cache_docwin_old.json").read_text(encoding="utf-8"))
    assert reuse["1"] == [[5, 500]]                     # cap p3 da co wins -> tai dung
    assert new["1"] == [[6, 600], [7, 700]]              # doc 1 = top-100, doc 2 = split -> loai
    assert k_new["1"] == [[6, 600, 0.0], [7, 700, 0.0]]  # 6,7 deu la doc moi
    assert k_old == {}


def test_cmd_dense_merge(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"1": [[1, 1, 0.9], [2, 2, 0.5]]}), encoding="utf-8")
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"1": [[3, 3, 0.7]]}), encoding="utf-8")
    args = type("A", (), {"cache_old": str(old), "cache_new": str(new), "top": 2,
                          "out": str(tmp_path / "out.json"), "report": ""})()
    assert vi23.cmd_dense_merge(args) == 0
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert [h[0] for h in out["1"]] == [1, 3]


def test_cmd_newonly30(tmp_path):
    fhyde = tmp_path / "f.json"
    fhyde.write_text(json.dumps({"1": [[11, 1, 1.0], [1, 2, 0.9], [12, 3, 0.8]],
                                "2": [[1, 1, 1.0], [13, 2, 0.9]]}), encoding="utf-8")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([11, 12, 13]), encoding="utf-8")
    args = type("A", (), {"fhyde": str(fhyde), "new_ids": str(ids), "n": 30,
                          "out": str(tmp_path / "sub.json"), "report": ""})()
    assert vi23.cmd_newonly30(args) == 0
    sub = json.loads((tmp_path / "sub.json").read_text(encoding="utf-8"))
    assert sub[0]["relevant_docs"] == [11, 12]
    assert sub[1]["relevant_docs"] == [13]
    assert all(s["relevant_chunks"] == [] for s in sub)
