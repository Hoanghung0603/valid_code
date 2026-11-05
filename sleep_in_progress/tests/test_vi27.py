import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi27
from cx93 import apply_bonus
from diag14 import strip_meta


def test_zip_specs_and_constants():
    assert vi27.ZIPS == [("vi27_b2", 2.0), ("vi27_b4", 4.0)]
    assert vi27.MAX_ZIP == 99_000_000
    assert vi27.DELTA == 57 and vi27.SANITY_DELTA == 57
    assert vi27.CANDS == vi27.C26 / "chunk_dw_u900_v25.json.zst"
    assert vi27.OUT == vi27.ROOT / "data/sub/diag27"
    assert vi27.REF_SUB == vi27.W / "submission_vi25_z99.json"
    assert vi27.REF_DOC == vi27.W / "doclist_vi25_z99.json"


def test_chain_args_change_only_b_cov():
    a2 = list(map(str, vi27.chain_args("vi27_b2", 57, 2.0)))
    a4 = list(map(str, vi27.chain_args("vi27_b4", 57, 4.0)))
    for args in (a2, a4):
        assert args[args.index("--pool") + 1] == str(vi27.W / "cache_pool_v25.json")
        assert args[args.index("--dense") + 1] == str(vi27.W / "dense_v25_top200.json")
        assert args[args.index("--hyde") + 1] == str(vi27.W / "hyde_v25_top200.json")
        assert args[args.index("--q8b") + 1] == str(vi27.Q8)
        assert args[args.index("--cands") + 1] == str(vi27.CANDS)
        assert args[args.index("--ref-kq") + 1] == str(vi27.REF_K)
        assert args[args.index("--ref-chunks") + 1] == str(vi27.REF_C)
        assert args[args.index("--delta") + 1] == "57"
        assert args[args.index("--out-dir") + 1] == str(vi27.OUT)
        assert args[args.index("--mode") + 1] == "that"
    assert a2[a2.index("--b-cov") + 1] == "2.0"
    assert a4[a4.index("--b-cov") + 1] == "4.0"
    diff = [(x, y) for x, y in zip(a2, a4) if x != y]
    assert diff == [("vi27_b2", "vi27_b4"), ("2.0", "4.0")]


def test_apply_bonus_higher_b_covers_more_docs_same_k():
    order = [{"doc_id": 10, "chunk_text": "A1", "s": 5.0},
             {"doc_id": 10, "chunk_text": "A2", "s": 4.9},
             {"doc_id": 10, "chunk_text": "A3", "s": 4.8},
             {"doc_id": 11, "chunk_text": "B1", "s": 3.0},
             {"doc_id": 12, "chunk_text": "C1", "s": 2.9}]
    k = 3
    sels = {b: strip_meta(apply_bonus(order, b), k) for b in (1.0, 2.0, 4.0)}
    assert all(len(sel) == k for sel in sels.values())
    covered = {b: len({c["doc_id"] for c in sel}) for b, sel in sels.items()}
    assert covered[1.0] == 1
    assert covered[2.0] == 2
    assert covered[4.0] == 3
    assert covered[1.0] < covered[2.0] < covered[4.0]
    for sel in sels.values():
        assert {c["chunk_text"] for c in sel} <= {c["chunk_text"] for c in order}


def test_apply_bonus_b2_b4_boost_first_chunk_per_doc():
    order = [{"doc_id": 10, "chunk_text": "A1", "s": 5.0},
             {"doc_id": 11, "chunk_text": "B1", "s": 4.0}]
    for b in (2.0, 4.0):
        got = apply_bonus(order, b)
        assert [c["chunk_text"] for c in got] == ["A1", "B1"]
        keys = [c["s"] + b for c in got]
        assert keys == sorted(keys, reverse=True)


def test_require_sanity_1200_of_1200_else_exit2():
    ok = {"ok": True, "so_cau": 1200, "cau_khop_doc": 1200, "cau_khop_chunk": 1200}
    vi27.require_sanity(ok)
    for bad in [{"ok": False, "so_cau": 1200, "cau_khop_doc": 1200, "cau_khop_chunk": 1199},
                {"ok": True, "so_cau": 1199, "cau_khop_doc": 1199, "cau_khop_chunk": 1199},
                {"ok": True, "so_cau": 1200, "cau_khop_doc": 1199, "cau_khop_chunk": 1200}]:
        with pytest.raises(SystemExit) as e:
            vi27.require_sanity(bad)
        assert e.value.code == 2


def test_rank_band_edges():
    assert vi27.rank_band(None) == "khong_co_hang"
    assert vi27.rank_band(1) == "1-50"
    assert vi27.rank_band(50) == "1-50"
    assert vi27.rank_band(51) == "51-100"
    assert vi27.rank_band(100) == "51-100"
    assert vi27.rank_band(101) == "101-165"
    assert vi27.rank_band(165) == "101-165"
    assert vi27.rank_band(166) == "166+"
    assert vi27.rank_band(5000) == "166+"


def test_fhyde_rank_maps_dedup_first_position(tmp_path):
    p = tmp_path / "cache_fhyde.json"
    p.write_text(json.dumps({"1": [[10, 0, 1.0], [11, 0, 0.9], [10, 5, 0.8], [12, 0, 0.7]]}),
                 encoding="utf-8")
    maps = vi27.fhyde_rank_maps(p)
    assert maps == {"1": {"10": 1, "11": 2, "12": 3}}


def test_sub_metrics_vs_z99_pairs_and_new_doc_bands():
    ref = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": [
        {"doc_id": 10, "chunk_text": "A"}, {"doc_id": 11, "chunk_text": "B"}]}]
    sub = [{"id": 1, "relevant_docs": [10, 12], "relevant_chunks": [
        {"doc_id": 10, "chunk_text": "A"}, {"doc_id": 12, "chunk_text": "C"}]}]
    ref_pairs = {"1": vi27.chunk_pairs(ref[0])}
    ref_docs = {"1": vi27.covered_docs(ref[0])}
    rank_maps = {"1": {"10": 1, "11": 2, "12": 80}}
    m = vi27.sub_metrics(sub, ref_pairs, ref_docs, rank_maps)
    assert m["tong_chunk"] == 2
    assert m["TB_chunk_cau"] == 2.0
    assert m["TB_doc_khac_nhau_co_chunk_cau"] == 2.0
    assert m["cap_cau_doc_co_chunk"] == 2
    assert m["TB_chunk_moi_doc_co_chunk"] == 1.0
    assert m["TB_chunk_khac_z99_cau"] == 1.0
    assert m["pct_chunk_trung_z99"] == 50.0
    assert m["so_cap_doc_moi_phu"] == 1
    assert m["hang_fhyde_doc_moi"] == {"1-50": 0, "51-100": 1, "101-165": 0,
                                       "166+": 0, "khong_co_hang": 0}
    self_m = vi27.sub_metrics(ref, ref_pairs, ref_docs, rank_maps)
    assert self_m["TB_chunk_khac_z99_cau"] == 0.0
    assert self_m["pct_chunk_trung_z99"] == 100.0
    assert self_m["so_cap_doc_moi_phu"] == 0


def test_sub_metrics_new_doc_without_rank_is_khong_co_hang():
    ref = [{"id": 1, "relevant_docs": [10], "relevant_chunks": [{"doc_id": 10, "chunk_text": "A"}]}]
    sub = [{"id": 1, "relevant_docs": [10, 99], "relevant_chunks": [
        {"doc_id": 10, "chunk_text": "A"}, {"doc_id": 99, "chunk_text": "Z"}]}]
    m = vi27.sub_metrics(sub, {"1": vi27.chunk_pairs(ref[0])}, {"1": vi27.covered_docs(ref[0])}, {})
    assert m["hang_fhyde_doc_moi"]["khong_co_hang"] == 1
    assert m["so_cap_doc_moi_phu"] == 1


def test_find_delta_steps_down_by_one(monkeypatch):
    calls = []

    def fake(name, b, delta, max_zip):
        calls.append((name, b, delta, max_zip))
        code = 0 if delta <= 56 else 3
        return {"code": code, "zip_bytes": 99_100_000 if code else 98_900_000,
                "json_bytes": 643_000_000, "tong_chunk": 163811, "delta": delta, "name": name}

    monkeypatch.setattr(vi27, "chain_zip_check", fake)
    d, attempts = vi27.find_delta("vi27_b2", 2.0, 99_000_000, 57)
    assert d == 56
    assert [c[2] for c in calls] == [57, 56]
    assert attempts == [{"delta": 57, "code": 3, "zip_bytes": 99_100_000},
                        {"delta": 56, "code": 0, "zip_bytes": 98_900_000}]


def test_find_delta_first_fit_returns_57(monkeypatch):
    monkeypatch.setattr(vi27, "chain_zip_check",
                        lambda *a: {"code": 0, "zip_bytes": 98_976_669, "json_bytes": 643_215_253,
                                    "tong_chunk": 163811, "delta": a[2], "name": a[0]})
    d, attempts = vi27.find_delta("vi27_b4", 4.0, 99_000_000, 57)
    assert d == 57
    assert attempts == [{"delta": 57, "code": 0, "zip_bytes": 98_976_669}]


def test_find_delta_content_mismatch_raises(monkeypatch):
    monkeypatch.setattr(vi27, "chain_zip_check",
                        lambda *a: {"code": 1, "zip_bytes": 1, "json_bytes": 1,
                                    "tong_chunk": 1, "delta": a[2], "name": a[0]})
    with pytest.raises(RuntimeError):
        vi27.find_delta("vi27_b2", 2.0, 99_000_000, 57)
