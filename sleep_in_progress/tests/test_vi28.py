import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi23 as core
import vi28


def write_zstd_jsonl(path, rows):
    import zstandard
    with path.open("wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            for r in rows:
                z.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))


def write_zstd_json(path, obj):
    import zstandard
    with path.open("wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            z.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def test_constants_and_paths():
    assert vi28.MAX_ZIP == 99_000_000
    assert vi28.DELTA0 == 57 and vi28.SANITY_DELTA == 57
    assert vi28.NAME == "vi28" and vi28.SANITY_NAME == "vi28_sanity"
    assert (vi28.THR, vi28.LO, vi28.HI, vi28.WD, vi28.B_COV) == (-2.0, 40, 400, 4.0, 1.0)
    assert vi28.OUT == vi28.ROOT / "data/sub/diag28"
    assert vi28.REF_SUB == vi28.W / "submission_vi25_z99.json"
    assert vi28.REF_K == vi28.ROOT / "data/sub/diag14/submission_c120in.json"
    assert vi28.REF_C == vi28.ROOT / "data/sub/diag11/chunks_rrf3d2_cf4.json"
    assert vi28.Q8 == vi28.ROOT / "data/sub/qwen8b/cache_qwen8b_sorted.json"
    assert vi28.BAD_TSV.name == "contamination_bad_ids_20261024.tsv"
    assert vi28.FP_TSV.name == "contamination_false_positive_20261024.tsv"
    assert vi28.PKG_TAGS == ["20261017", "20261021", "20261023"]


def test_read_ids_tsv_skips_header_and_blank_lines(tmp_path):
    p = tmp_path / "ids.tsv"
    p.write_text("id\tshard\treasons\n\n101\trecov-0\tx\n202\trecov-1\ty\n", encoding="utf-8")
    assert vi28.read_ids_tsv(p) == [101, 202]


def test_read_ids_tsv_rejects_garbage_row(tmp_path):
    p = tmp_path / "ids.tsv"
    p.write_text("id\n101\nabc\n", encoding="utf-8")
    with pytest.raises(ValueError):
        vi28.read_ids_tsv(p)


def test_scan_package_id_and_text_sha(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    write_zstd_jsonl(pkg / "vi-recov-00000.jsonl.zst",
                     [{"id": 1, "text": "abc"}, {"id": 2, "text": None}])
    write_zstd_jsonl(pkg / "vi-recov-00001.jsonl.zst", [{"id": 3, "text": "abc"}])
    got = vi28.scan_package(pkg)
    assert got["records"] == 3
    assert set(got["ids"]) == {1, 2, 3}
    assert got["ids"][1] == got["ids"][3]  # cung text -> cung sha256
    assert got["ids"][2] == vi28.hashlib.sha256(b"").hexdigest()
    assert got["files"] == ["vi-recov-00000.jsonl.zst", "vi-recov-00001.jsonl.zst"]
    assert got["trung_id_khac_text"] == 0


def test_hypothesis_report_ok_when_fix_subset_same_text():
    old = {"a": {1: "h1", 2: "h2", 3: "h3"}, "b": {4: "h4", 5: "h5"}}
    fix = {"a": {2: "h2", 3: "h3"}, "b": {5: "h5"}}
    rep = vi28.hypothesis_report({1, 4}, set(), old, fix, {1, 2, 3, 4, 5, 6})
    assert rep["so_dong_bad"] == 2
    assert rep["tong"]["bad_trong_cu"] == 2
    assert rep["tong"]["bad_trong_fix"] == 0
    assert rep["tong"]["fix_ngoai_cu"] == 0
    assert rep["tong"]["cu_thua_sau_fix"] == 0
    assert rep["tong"]["text_hash_khac"] == 0
    assert rep["bad_trong_idx_vi25"] == 2
    assert rep["fix_ngoai_cu_gop"] == 0 and rep["text_hash_khac_gop"] == 0
    assert rep["ok_thuyet"] is True


def test_hypothesis_report_counts_canonical_dup_gap():
    old = {"a": {1: "h1", 2: "h2", 3: "h3", 4: "h4"}}
    fix = {"a": {2: "h2"}}
    rep = vi28.hypothesis_report({1}, set(), old, fix, set())
    assert rep["tong"]["cu_thua_sau_fix"] == 2  # 3, 4: trung canonical khong gỡ
    assert rep["cu_thua_sau_fix_gop"] == 2
    assert rep["ok_thuyet"] is True


def test_hypothesis_report_stops_on_fix_id_outside_or_text_change():
    base_old = {"a": {1: "h1", 2: "h2"}}
    new_id = vi28.hypothesis_report(set(), set(), base_old, {"a": {2: "h2", 9: "h9"}}, set())
    assert new_id["tong"]["fix_ngoai_cu"] == 1 and new_id["ok_thuyet"] is False
    text = vi28.hypothesis_report(set(), set(), base_old, {"a": {2: "hX"}}, set())
    assert text["tong"]["text_hash_khac"] == 1 and text["ok_thuyet"] is False


def test_hypothesis_report_fp_not_removed():
    rep = vi28.hypothesis_report({1}, {2, 3}, {"a": {1: "h"}}, {"a": {}}, set())
    assert rep["so_dong_fp"] == 2 and rep["fp_trong_bad"] == 0


def test_impact_report_counts_pairs_chunks_and_refs():
    sub = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": [
            {"doc_id": 10, "chunk_text": "A"}, {"doc_id": 10, "chunk_text": "B"},
            {"doc_id": 11, "chunk_text": "C"}]},
           {"id": 2, "relevant_docs": [20], "relevant_chunks": [
            {"doc_id": 20, "chunk_text": "D"}]}]
    ref_chunks = [{"id": 1, "relevant_chunks": [{"doc_id": 10, "chunk_text": "A"},
                                                {"doc_id": 99, "chunk_text": "Z"}]}]
    c120in = [{"id": 1, "relevant_docs": [10, 99], "relevant_chunks": [
                {"doc_id": 10, "chunk_text": "A"}, {"doc_id": 77, "chunk_text": "Y"}]}]
    got = vi28.impact_report(sub, {10, 77}, ref_chunks, c120in)
    assert got["so_cap_qdoc_bad_trong_bai"] == 1
    assert got["TB_qdoc_bad_cau"] == 0.5
    assert got["so_cau_anh_huong"] == 1
    assert got["so_chunk_cua_doc_bad"] == 2
    assert got["ref_chunks"] == {"so_cap_qdoc_bad": 1, "so_id_bad": 1}
    assert got["c120in"] == {"so_cap_qdoc_bad": 2, "so_id_bad": 2}


def test_filter_hits_preserves_order_and_scores():
    hits = [[1, 0, 0.9], [2, 1, 0.8], [1, 2, 0.7], [3, 3, None]]
    assert vi28.filter_hits(hits, {"1", "9"}) == [[2, 1, 0.8], [3, 3, None]]
    assert vi28.filter_hits([], {"1"}) == []
    assert vi28.filter_hits(hits, set()) == hits


def test_filter_cache_filter_cands_filter_ref_chunks():
    cache = {"1": [[10, 0, 1.0], [11, 1, 0.5]], "2": [[10, 0, 0.2]]}
    assert vi28.filter_cache(cache, {"10"}) == {"1": [[11, 1, 0.5]], "2": []}
    cands = {"1": [[10, 0, 0, 2.5, "x"], [11, 1, 0, 2.0, "y"]]}
    assert vi28.filter_cands(cands, {"10"}) == {"1": [[11, 1, 0, 2.0, "y"]]}
    ref = [{"id": 1, "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"},
                                         {"doc_id": 11, "chunk_text": "y"}]},
           {"id": 2, "relevant_chunks": [{"doc_id": 10, "chunk_text": "z"}]}]
    got = vi28.filter_ref_chunks(ref, {"10"})
    assert got == [{"id": 1, "relevant_chunks": [{"doc_id": 11, "chunk_text": "y"}]},
                   {"id": 2, "relevant_chunks": []}]


def test_write_inputs_filters_all_six_and_keeps_rest(tmp_path, monkeypatch):
    w = tmp_path / "diag25"
    w.mkdir()
    (w / "cache_pool_v25.json").write_text(json.dumps({"1": [[10, 0, 0.9], [11, 1, None]]}),
                                           encoding="utf-8")
    (w / "dense_v25_top200.json").write_text(json.dumps({"1": [[10, 0, 0.5]]}), encoding="utf-8")
    (w / "hyde_v25_top200.json").write_text(json.dumps({"1": [[11, 1, 0.4]]}), encoding="utf-8")
    write_zstd_json(w / "chunk_dw_u900_v25.json.zst",
                    {"1": [[10, 0, 0, 2.5, "x"], [11, 1, 0, 2.0, "y"]]})
    q8 = tmp_path / "q8.json"
    q8.write_text(json.dumps({"1": [[10, 0, 3.0]]}), encoding="utf-8")
    ref_c = tmp_path / "cf4.json"
    ref_c.write_text(json.dumps([{"id": 1, "relevant_chunks": [
        {"doc_id": 10, "chunk_text": "x"}, {"doc_id": 11, "chunk_text": "y"}]}]),
        encoding="utf-8")
    monkeypatch.setattr(vi28, "W", w)
    monkeypatch.setattr(vi28, "Q8", q8)
    monkeypatch.setattr(vi28, "REF_C", ref_c)
    indir = tmp_path / "in1"
    rep = vi28.write_inputs({10}, indir)
    f = rep["files"]
    assert rep["bad_id"] == 1
    assert (f["pool_v28.json"]["truoc"], f["pool_v28.json"]["sau"], f["pool_v28.json"]["bo"]) == (2, 1, 1)
    assert f["q8b_v28.json"]["bo"] == 1
    assert f["dense_v28.json"]["sau"] == 0
    assert f["hyde_v28.json"]["bo"] == 0
    assert f["cands_v28.json.zst"]["bo"] == 1
    assert f["ref_chunks_v28.json"]["bo"] == 1
    assert core.load_json(indir / "pool_v28.json") == {"1": [[11, 1, None]]}
    assert core.load_json(indir / "q8b_v28.json") == {"1": []}
    assert core.load_json(indir / "cands_v28.json.zst") == {"1": [[11, 1, 0, 2.0, "y"]]}
    assert core.load_json(indir / "ref_chunks_v28.json") == [
        {"id": 1, "relevant_chunks": [{"doc_id": 11, "chunk_text": "y"}]}]


def test_chain_args_use_filtered_inputs_and_vi25_params(tmp_path):
    args = list(map(str, vi28.chain_args("vi28", 57, tmp_path)))
    assert args[args.index("--pool") + 1] == str(tmp_path / "pool_v28.json")
    assert args[args.index("--q8b") + 1] == str(tmp_path / "q8b_v28.json")
    assert args[args.index("--dense") + 1] == str(tmp_path / "dense_v28.json")
    assert args[args.index("--hyde") + 1] == str(tmp_path / "hyde_v28.json")
    assert args[args.index("--cands") + 1] == str(tmp_path / "cands_v28.json.zst")
    assert args[args.index("--ref-kq") + 1] == str(vi28.REF_K)
    assert args[args.index("--ref-chunks") + 1] == str(tmp_path / "ref_chunks_v28.json")
    assert args[args.index("--thr") + 1] == "-2.0"
    assert args[args.index("--lo") + 1] == "40"
    assert args[args.index("--hi") + 1] == "400"
    assert args[args.index("--wd") + 1] == "4.0"
    assert args[args.index("--b-cov") + 1] == "1.0"
    assert args[args.index("--delta") + 1] == "57"
    assert args[args.index("--out-dir") + 1] == str(vi28.OUT)
    assert args[args.index("--mode") + 1] == "that"


def test_require_sanity_1200_of_1200_else_exit2():
    vi28.require_sanity({"ok": True, "so_cau": 1200, "cau_khop_doc": 1200, "cau_khop_chunk": 1200})
    for bad in [{"ok": False, "so_cau": 1200, "cau_khop_doc": 1200, "cau_khop_chunk": 1200},
                {"ok": True, "so_cau": 1200, "cau_khop_doc": 1199, "cau_khop_chunk": 1200},
                {"ok": True, "so_cau": 1199, "cau_khop_doc": 1199, "cau_khop_chunk": 1199}]:
        with pytest.raises(SystemExit) as e:
            vi28.require_sanity(bad)
        assert e.value.code == 2


def fake_chain(max_delta_fit):
    def run(name, delta, max_zip, indir, validate=False):
        zb = 90_000_000 + 250_000 * delta
        fits = delta <= max_delta_fit
        return {"code": 0 if fits else 3,
                "next_delta": None if fits else max(max_delta_fit, 0),
                "zip_bytes": zb, "json_bytes": 1_000_000 * (100 + delta),
                "tong_chunk": 1200 * (100 + delta), "hist_kcap": [[100, 300, 1200]],
                "n_q_TB": 100.0}
    return run


def test_find_delta_picks_largest_with_real_zip_under_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(vi28, "OUT", tmp_path)
    monkeypatch.setattr(vi28, "chain_zip_check", fake_chain(max_delta_fit=52))
    d, attempts = vi28.find_delta(tmp_path, 99_000_000, 48, "probe")
    assert d == 52
    assert attempts[-1]["delta"] >= 52
    assert all(a["code"] == 0 for a in attempts if a["delta"] <= 52)


def test_find_delta_decreases_when_start_over_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(vi28, "OUT", tmp_path)
    monkeypatch.setattr(vi28, "chain_zip_check", fake_chain(max_delta_fit=30))
    d, _attempts = vi28.find_delta(tmp_path, 99_000_000, 60, "probe")
    assert d == 30


def test_find_delta_stops_when_chunk_count_saturates(tmp_path, monkeypatch):
    monkeypatch.setattr(vi28, "OUT", tmp_path)

    def run(name, delta, max_zip, indir, validate=False):
        return {"code": 0, "next_delta": None, "zip_bytes": 90_000_000,
                "json_bytes": 600_000_000, "tong_chunk": 200000,
                "hist_kcap": [[100, 120, 1200]], "n_q_TB": 100.0}

    monkeypatch.setattr(vi28, "chain_zip_check", run)
    d, attempts = vi28.find_delta(tmp_path, 99_000_000, 5, "probe")
    assert d == 5
    assert len(attempts) <= 4


def test_find_delta_resumes_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vi28, "OUT", tmp_path)
    calls = []
    monkeypatch.setattr(vi28, "chain_zip_check", fake_chain(max_delta_fit=52))
    vi28.find_delta(tmp_path, 99_000_000, 52, "probe")

    def counted(name, delta, max_zip, indir, validate=False):
        calls.append(delta)
        return fake_chain(52)(name, delta, max_zip, indir, validate)

    monkeypatch.setattr(vi28, "chain_zip_check", counted)
    d, _ = vi28.find_delta(tmp_path, 99_000_000, 52, "probe")
    assert d == 52
    assert 52 not in calls  # da cache tu lan truoc


def test_assert_clean_passes_only_without_bad_and_off_doclist():
    sub = [{"id": i, "relevant_docs": [10, 11],
            "relevant_chunks": [{"doc_id": 10, "chunk_text": "a"}]} for i in range(1, 1201)]
    assert vi28.assert_clean(sub, {99})["bad_doc"] == 0
    bad_doc = [dict(sub[0], relevant_docs=[99, 11])]
    with pytest.raises(AssertionError):
        vi28.assert_clean(bad_doc + sub[1:], {99})
    off = [{"id": 1, "relevant_docs": [10], "relevant_chunks": [{"doc_id": 11, "chunk_text": "a"}]}]
    with pytest.raises(AssertionError):
        vi28.assert_clean(off + sub[1:], set())
    short = sub[:1199]
    with pytest.raises(AssertionError):
        vi28.assert_clean(short, set())


def test_sub_row_metrics():
    sub = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": [
            {"doc_id": 10, "chunk_text": "a"}, {"doc_id": 11, "chunk_text": "b"}]}]
    row = vi28.sub_row(sub, {"n_q_TB": 164.5}, {"bytes": 5, "sha256": "ab"}, 57)
    assert row["so_cau"] == 1 and row["delta"] == 57
    assert row["TB_doc_cau"] == 2.0 and row["TB_chunk_cau"] == 2.0
    assert row["TB_doc_co_chunk_cau"] == 2.0 and row["n_q_TB"] == 164.5
    assert row["zip_bytes"] == 5 and row["sha256"] == "ab"
    assert row["tong_chunk"] == 2
