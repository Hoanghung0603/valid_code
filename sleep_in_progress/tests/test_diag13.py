import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from diag13 import (build_chunkceil, build_docceil, build_nqf, calibrate_fused_thr,  # noqa: E402
                    dedup_chunk_rows, job1_stats, job2_zips, order_pool_docs, pick_nq_fused,
                    rankdata, spearman, with_chunk_docs)


def test_rankdata_ties_get_average_rank():
    assert rankdata([10, 20, 20, 30]) == [1, 2.5, 2.5, 4]
    assert rankdata([5]) == [1]


def test_spearman_perfect_reverse_and_constant():
    assert abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9
    assert abs(spearman([1, 2, 3], [1, 4, 9]) - 1.0) < 1e-9
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_pick_nq_fused_ratio_threshold_and_clamp():
    hits = [[1, 0, 1.0], [2, 0, 0.5], [3, 0, 0.25], [4, 0, 0.1]]
    assert pick_nq_fused(hits, 0.2, lo=0, hi=10, method="ratio") == 3
    assert pick_nq_fused(hits, 0.2, lo=4, hi=10, method="ratio") == 4
    assert pick_nq_fused(hits, 0.2, lo=0, hi=2, method="ratio") == 2
    assert pick_nq_fused([], 0.2, lo=7, hi=10) == 7


def test_pick_nq_fused_zscore_and_none_ignored():
    hits = [[1, 0, 0.0], [2, 0, 1.0], [3, 0, 2.0], [4, 0, 3.0], [5, 0, None]]
    assert pick_nq_fused(hits, 0.0, lo=0, hi=10, method="zscore") == 2
    try:
        pick_nq_fused(hits, 0.0, method="la")
        assert False
    except ValueError:
        pass


def test_calibrate_fused_thr_hits_target_mean():
    per_q = [[[i, 0, float(100 - i)] for i in range(100)] for _ in range(8)]
    thr, ns = calibrate_fused_thr(per_q, "ratio", target=50.0, lo=0, hi=100)
    assert abs(sum(ns) / len(ns) - 50.0) <= 2
    thr2, ns2 = calibrate_fused_thr(per_q, "zscore", target=50.0, lo=0, hi=100)
    assert abs(sum(ns2) / len(ns2) - 50.0) <= 2
    assert thr2 != thr


def test_order_pool_docs_fhyde_first_then_pool():
    pool = [[9, 0, 3.0], [8, 0, 2.0], [7, 0, 1.0], [6, 0, 0.0]]
    fhyde = [[7, 0, 0.5], [5, 0, 0.4], [9, 0, 0.3]]
    assert order_pool_docs(pool, fhyde) == [7, 9, 8, 6]


def test_with_chunk_docs_adds_missing_in_order():
    docs, add = with_chunk_docs([1, 2], [{"doc_id": 2, "chunk_text": "a"},
                                         {"doc_id": 5, "chunk_text": "b"},
                                         {"doc_id": 5, "chunk_text": "c"}])
    assert docs == [1, 2, 5] and add == 1


def test_dedup_chunk_rows_keeps_highest_logit_and_k_cut():
    rows = [[1, 1, 0, 0.5, "A"], [2, 1, 1, 0.9, "B"], [3, 1, 2, 0.7, "A"]]
    assert dedup_chunk_rows(rows) == [{"doc_id": 2, "chunk_text": "B"}, {"doc_id": 3, "chunk_text": "A"}]
    assert dedup_chunk_rows(rows, k=1) == [{"doc_id": 2, "chunk_text": "B"}]


def test_build_docceil_full_pool_order_and_chunk_docs():
    pool = {"1": [[5, 0, 1.0], [6, 0, 0.5], [7, 0, 0.0]]}
    fhyde = {"1": [[7, 0, 0.9], [6, 0, 0.8]]}
    chunks = [{"doc_id": 9, "chunk_text": "x"}, {"doc_id": 5, "chunk_text": "y"}]
    sub = [{"id": 1, "relevant_docs": [6], "relevant_chunks": chunks}]
    out, added = build_docceil(sub, pool, fhyde)
    assert out[0]["relevant_docs"] == [7, 6, 5, 9]
    assert out[0]["relevant_chunks"] is chunks
    assert set(out[0]) == {"id", "relevant_docs", "relevant_chunks"}
    assert added == [1]


def test_build_chunkceil_even_odd_dedup_and_added_docs():
    rows2 = [[10, 1, 0, 0.5, "a"], [11, 2, 1, 0.9, "b"], [11, 2, 2, 0.7, "b"]]
    rows4 = [[20, 1, 0, 1.0, "c"]]
    sub = [{"id": 1, "relevant_docs": [1], "relevant_chunks": [{"doc_id": 1, "chunk_text": "z"}]},
           {"id": 2, "relevant_docs": [1], "relevant_chunks": []},
           {"id": 3, "relevant_docs": [2], "relevant_chunks": []},
           {"id": 4, "relevant_docs": [3], "relevant_chunks": []}]
    out, st = build_chunkceil(sub, {"2": rows2, "4": rows4})
    assert [c["chunk_text"] for c in out[1]["relevant_chunks"]] == ["b", "a"]
    assert out[0]["relevant_chunks"] == []
    assert out[0]["relevant_docs"] == [1]
    assert out[3]["relevant_chunks"] == [{"doc_id": 20, "chunk_text": "c"}]
    assert out[1]["relevant_docs"] == [1, 11, 10]
    assert out[3]["relevant_docs"] == [3, 20]
    assert st["cau_chan"] == 2 and st["cau_le"] == 2
    assert st["TB_chunk_nua_chan"] == 1.5 and st["chunk_trung_bo"] == 1
    assert st["tong_doc_them"] == 3
    out_k, st_k = build_chunkceil(sub, {"2": rows2, "4": rows4}, k=1)
    assert [c["chunk_text"] for c in out_k[1]["relevant_chunks"]] == ["b"]
    assert st_k["TB_chunk_nua_chan"] == 1.0


def test_build_nqf_prefix_fhyde_and_chunk_docs():
    fhyde = {"1": [[7, 0, 1.0], [6, 0, 0.6], [5, 0, 0.3], [4, 0, 0.05]]}
    pool = {"1": [[5, 0, 1.0], [6, 0, 0.5], [7, 0, 0.0], [4, 0, -1.0]]}
    chunks = [{"doc_id": 9, "chunk_text": "x"}]
    sub = [{"id": 1, "relevant_docs": [7, 6], "relevant_chunks": chunks}]
    out, ns, added = build_nqf(sub, pool, fhyde, thr=0.2, method="ratio", lo=1, hi=10)
    assert ns == [3]
    assert out[0]["relevant_docs"] == [7, 6, 5, 9]
    assert out[0]["relevant_chunks"] is chunks
    assert added == [1]


def test_job1_stats_bands_and_outside():
    pool = {"1": [[i, 0, 1.0] for i in range(1, 106)]}
    fhyde = {"1": [[100 + i, 0, 1.0] for i in range(1, 6)]}
    docs = list(range(1, 106))
    chunks = [{"doc_id": 101, "chunk_text": "a"}, {"doc_id": 2, "chunk_text": "b"}]
    sub = [{"id": 1, "relevant_docs": docs, "relevant_chunks": chunks}]
    rep = job1_stats(sub, pool, fhyde)
    assert rep["cau"] == 1
    assert rep["ngoai_top100_bge"]["TB"] == 5
    assert rep["ngoai_top100_bge"]["trung_vi"] == 5
    assert rep["theo_hang"]["1-30"]["TB_ngoai_top100"] == 0
    assert rep["theo_hang"]["101+"]["TB_ngoai_top100"] == 5
    assert rep["theo_hang"]["101+"]["TB_doc_co_chunk"] == 1
    assert rep["theo_hang"]["1-30"]["TB_doc_co_chunk"] == 1
    assert rep["fhyde_top100_ngoai_top100_bge"]["TB"] == 5
    assert rep["n_q_cu_logit_bge"]["TB"] == 105


def test_job2_zips_roundtrip_on_tiny(tmp_path):
    sub = [{"id": 1, "relevant_docs": [1, 2],
            "relevant_chunks": [{"doc_id": 1, "chunk_text": "alo xin chao"}]},
           {"id": 2, "relevant_docs": [], "relevant_chunks": []}]
    src = tmp_path / "submission_split.json"
    src.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    rep = job2_zips(str(src), str(tmp_path))
    assert rep["json_bytes"] == src.stat().st_size
    assert [r["method"] for r in rep["zip"]] == ["ZIP_DEFLATED", "ZIP_BZIP2", "ZIP_LZMA"]
    for r in rep["zip"]:
        assert r["doc_lai"] == "OK"
        assert r["bang_goc_bytes"] and r["bang_goc_json"]
        assert (tmp_path / r["file"]).stat().st_size == r["bytes"]
        assert r["bytes"] > 0
