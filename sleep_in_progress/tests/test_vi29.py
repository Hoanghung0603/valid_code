import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi23 as core
import vi29


def write_zstd_json(path, obj):
    import zstandard
    with path.open("wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            z.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def toy_inputs():
    """Du lieu do choi: 1 cau, 4 doc, chunk "chunkA" xuat hien o 2 doc khac nhau."""
    pool = {"1": [[10, 0, 1.0], [11, 1, 0.5], [12, 2, 0.2], [13, 3, None]]}
    q8b = {"1": [[11, 0, 2.0], [10, 1, 0.1]]}
    dense = {"1": [[12, 0, 0.3], [10, 1, 0.2]]}
    hyde = {"1": [[10, 0, 0.3]]}
    cands = {"1": [[10, 1, 0, 5.0, "chunkA"],
                   [11, 2, 0, 4.5, "chunkA"],
                   [11, 2, 5, 0.2, "chunkA"],   # trung (doc_id, text) voi dong tren
                   [12, 3, 0, 1.0, "chunkB"]]}
    kq = {"1": 2}
    ref_chunks = {"1": [{"doc_id": 11, "chunk_text": "chunkA"},
                        {"doc_id": 12, "chunk_text": "chunkB"}]}
    return pool, q8b, dense, hyde, cands, kq, ref_chunks


def pairs_of(sel):
    return [(str(c["doc_id"]), c["chunk_text"]) for c in sel]


def test_constants_and_paths():
    assert vi29.MAX_ZIP == 99_000_000
    assert vi29.DELTA0 == 57 and vi29.SANITY_DELTA == 57
    assert vi29.NAME == "vi29" and vi29.SANITY_NAME == "vi29_sanity"
    assert (vi29.THR, vi29.LO, vi29.HI, vi29.WD, vi29.B_COV) == (-2.0, 40, 400, 4.0, 1.0)
    assert vi29.OUT == vi29.ROOT / "data/sub/diag29"
    assert vi29.IN1_28 == vi29.ROOT / "data/sub/diag28/in1"
    assert vi29.SUB28 == vi29.ROOT / "data/sub/diag28/submission_vi28.json"
    assert vi29.CORPUS.name == "links_corpus_vi_en_zh.parquet"
    assert vi29.REF_K == vi29.ROOT / "data/sub/diag14/submission_c120in.json"
    assert vi29.REF_C == vi29.ROOT / "data/sub/diag11/chunks_rrf3d2_cf4.json"
    assert (vi29.WINDOW_WORDS, vi29.WINDOW_STRIDE, vi29.MAX_LEN) == (300, 250, 512)
    assert vi29.TOK_SEED == 2929 and vi29.TOK_N_Q == 300 and vi29.TOK_N_DOC == 10


def test_order_q_docdedup_giu_trung_khac_doc_bo_trung_cung_doc():
    rows = [[10, 1, 0, 5.0, "x"], [11, 2, 0, 3.0, "x"], [10, 1, 1, 1.0, "x"],
            [12, 3, 0, 2.0, "y"]]
    out = vi29.order_q_docdedup(rows, [], allowed={"10", "11", "12"},
                                dedup_mode="doc_text", wd=4.0)
    pr = pairs_of(out)
    assert pr.count(("10", "x")) == 1          # trung (doc_id, text) cung doc -> giu 1
    assert ("11", "x") in pr                   # cung text khac doc -> giu lai
    assert ("12", "y") in pr
    assert len(pr) == 3


def test_order_q_docdedup_text_mode_giong_order_q_cu():
    from cx93 import filter_base
    from diag14 import order_q
    rows = [[10, 1, 0, 5.0, "x"], [11, 2, 0, 3.0, "x"], [12, 3, 0, 2.0, "y"]]
    fused = [[10, 0, 0.5], [11, 1, 0.4], [12, 2, 0.1]]
    allowed = {"10", "11", "12"}
    assert vi29.order_q_docdedup(rows, fused, allowed, "text", 4.0) == \
        filter_base(order_q(rows, fused, 4.0), allowed)
    assert vi29.order_q_docdedup(rows, fused, allowed, "text", 4.0) != \
        vi29.order_q_docdedup(rows, fused, allowed, "doc_text", 4.0)


def test_order_q_docdedup_loc_doclist_truoc_khi_bo_trung():
    # "x" cua doc 99 (ngoai doclist) co s cao hon doc 10: luat cu loai luon "x" cua doc 10
    rows = [[99, 1, 0, 9.0, "x"], [10, 2, 0, 1.0, "x"]]
    fused = [[99, 0, 1.0], [10, 1, 0.5]]
    allowed = {"10"}
    cu = vi29.order_q_docdedup(rows, fused, allowed, "text", 4.0)
    moi = vi29.order_q_docdedup(rows, fused, allowed, "doc_text", 4.0)
    assert pairs_of(cu) == []
    assert pairs_of(moi) == [("10", "x")]


def test_order_q_docdedup_zc_tinh_tren_moi_dong_ung_vien():
    from chunk_docfuse import zscores
    rows = [[10, 1, 0, 5.0, "x"], [99, 2, 0, 0.0, "z"], [11, 3, 0, 1.0, "w"]]
    out = vi29.order_q_docdedup(rows, [], allowed={"10"}, dedup_mode="doc_text", wd=0.0)
    assert pairs_of(out) == [("10", "x")]
    assert out[0]["zc"] == pytest.approx(zscores([5.0, 0.0, 1.0])[0])


def test_order_q_docdedup_bat_dedup_mode_khac():
    with pytest.raises(ValueError):
        vi29.order_q_docdedup([], [], None, "text_only")


def test_chain_build_v29_text_tai_lap_chain_build_cu():
    pool, q8b, dense, hyde, cands, kq, ref = toy_inputs()
    rrf_cu, fhyde_cu, sub_cu, rep_cu = core.chain_build(
        pool, q8b, dense, hyde, cands, kq, ref, -2.0, 40, 400, 4.0, 1.0, 57)
    rrf_moi, fhyde_moi, sub_moi, rep_moi = vi29.chain_build_v29(
        pool, q8b, dense, hyde, cands, kq, ref, -2.0, 40, 400, 4.0, 1.0, 57,
        dedup_mode="text")
    assert sub_moi == sub_cu
    assert rrf_moi == rrf_cu and fhyde_moi == fhyde_cu
    assert {k: v for k, v in rep_moi.items()
            if k not in ("dedup_mode", "nq_mode", "nq_fused_thr")} == rep_cu
    # luat cu: "chunkA" chi con 1 ban (khong phai 2 doc)
    assert sum(1 for c in sub_cu[0]["relevant_chunks"] if c["chunk_text"] == "chunkA") == 1


def test_chain_build_v29_doc_text_giu_chunk_trung_text_khac_doc():
    pool, q8b, dense, hyde, cands, kq, ref = toy_inputs()
    _rrf, _fh, sub, rep = vi29.chain_build_v29(
        pool, q8b, dense, hyde, cands, kq, ref, -2.0, 40, 400, 4.0, 1.0, 57,
        dedup_mode="doc_text")
    pr = pairs_of(sub[0]["relevant_chunks"])
    assert pr.count(("10", "chunkA")) == 1
    assert pr.count(("11", "chunkA")) == 1      # doc khac, cung text -> van con
    assert pr.count(("11", "chunkA")) + pr.count(("10", "chunkA")) == 2
    assert ("12", "chunkB") in pr
    assert len(pr) == 3                          # dong trung (11, chunkA) da bi bo
    assert rep["dedup_mode"] == "doc_text"
    assert rep["tong_chunk"] == 3


def test_chain_build_v29_bat_dedup_mode_khac():
    pool, q8b, dense, hyde, cands, kq, ref = toy_inputs()
    with pytest.raises(ValueError):
        vi29.chain_build_v29(pool, q8b, dense, hyde, cands, kq, ref, dedup_mode="xx")


def test_filter_helpers_giu_thu_tu_khong_lap_cho():
    cache = {"1": [[10, 0, 0.9], [11, 1, 0.5], [12, 2, None]]}
    assert vi29.filter_cache(cache, {11}, {10, 12}) == {"1": [[10, 0, 0.9], [12, 2, None]]}
    assert vi29.filter_cache(cache, set(), {10}) == {"1": [[10, 0, 0.9]]}
    cands = {"1": [[10, 0, 0, 2.5, "x"], [11, 1, 0, 2.0, "y"], [12, 2, 0, 1.0, "z"]]}
    assert vi29.filter_cands(cands, {11}, {10, 12}) == \
        {"1": [[10, 0, 0, 2.5, "x"], [12, 2, 0, 1.0, "z"]]}
    ref = [{"id": 1, "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"},
                                         {"doc_id": 11, "chunk_text": "y"}]}]
    assert vi29.filter_ref_chunks(ref, set(), {10}) == \
        [{"id": 1, "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"}]}]


def test_write_inputs_loc_bad_va_ngoai_kho(tmp_path, monkeypatch):
    w = tmp_path / "diag25"
    w.mkdir()
    (w / "cache_pool_v25.json").write_text(json.dumps({"1": [[10, 0, 0.9], [11, 1, None],
                                                             [12, 2, 0.5]]}), encoding="utf-8")
    (w / "dense_v25_top200.json").write_text(json.dumps({"1": [[10, 0, 0.5], [12, 1, 0.4]]}),
                                             encoding="utf-8")
    (w / "hyde_v25_top200.json").write_text(json.dumps({"1": [[11, 1, 0.4]]}), encoding="utf-8")
    write_zstd_json(w / "chunk_dw_u900_v25.json.zst",
                    {"1": [[10, 0, 0, 2.5, "x"], [11, 1, 0, 2.0, "y"], [12, 2, 0, 1.0, "z"]]})
    q8 = tmp_path / "q8.json"
    q8.write_text(json.dumps({"1": [[10, 0, 3.0], [12, 1, 1.0]]}), encoding="utf-8")
    ref_c = tmp_path / "cf4.json"
    ref_c.write_text(json.dumps([{"id": 1, "relevant_chunks": [
        {"doc_id": 10, "chunk_text": "x"}, {"doc_id": 11, "chunk_text": "y"},
        {"doc_id": 12, "chunk_text": "z"}]}]), encoding="utf-8")
    monkeypatch.setattr(vi29, "W", w)
    monkeypatch.setattr(vi29, "Q8", q8)
    monkeypatch.setattr(vi29, "REF_C", ref_c)
    indir = tmp_path / "in1"
    rep = vi29.write_inputs({11}, indir, {10, 13})   # bad = 11; kho chuan = {10, 13}
    f = rep["files"]
    assert rep["bad_id"] == 1 and rep["co_loc_ngoai_kho"] is True
    # 12 ngoai kho bi loai, 11 bad bi loai, 10 giu
    assert (f["pool_v29.json"]["truoc"], f["pool_v29.json"]["sau"], f["pool_v29.json"]["bo"]) == (3, 1, 2)
    assert f["q8b_v29.json"]["sau"] == 1
    assert f["dense_v29.json"]["sau"] == 1
    assert f["hyde_v29.json"]["sau"] == 0
    assert f["cands_v29.json.zst"]["sau"] == 1
    assert f["ref_chunks_v29.json"]["sau"] == 1
    assert core.load_json(indir / "pool_v29.json") == {"1": [[10, 0, 0.9]]}
    assert core.load_json(indir / "cands_v29.json.zst") == {"1": [[10, 0, 0, 2.5, "x"]]}
    assert core.load_json(indir / "ref_chunks_v29.json") == [
        {"id": 1, "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"}]}]
    # khong co corpus: chi loc bad (giong vi28)
    rep2 = vi29.write_inputs({11}, tmp_path / "in0", None)
    assert rep2["co_loc_ngoai_kho"] is False
    assert rep2["files"]["pool_v29.json"]["sau"] == 2


def test_assert_clean_bat_bad_ngoai_kho_trung_cap():
    sub = [{"id": i, "relevant_docs": [10, 11],
            "relevant_chunks": [{"doc_id": 10, "chunk_text": "a"},
                                {"doc_id": 11, "chunk_text": "b"}]} for i in range(1, 1201)]
    got = vi29.assert_clean(sub, {99}, {10, 11})
    assert got["cau"] == 1200 and got["cap_doc_text_trung"] == 0
    with pytest.raises(AssertionError):
        vi29.assert_clean(sub, {10}, {10, 11})                       # bad trong docs/chunks
    with pytest.raises(AssertionError):
        vi29.assert_clean(sub, set(), {10})                          # 11 ngoai kho
    trung = [dict(sub[0], relevant_chunks=[{"doc_id": 10, "chunk_text": "a"},
                                           {"doc_id": 10, "chunk_text": "a"}])]
    with pytest.raises(AssertionError):
        vi29.assert_clean(trung + sub[1:], set(), {10, 11})          # trung (doc_id, text)
    khac_doc = [dict(sub[0], relevant_chunks=[{"doc_id": 10, "chunk_text": "a"},
                                              {"doc_id": 11, "chunk_text": "a"}])]
    assert vi29.assert_clean(khac_doc + sub[1:], set(), {10, 11})["cap_doc_text_trung"] == 0
    off = [dict(sub[0], relevant_docs=[10],
                relevant_chunks=[{"doc_id": 11, "chunk_text": "a"}])]
    with pytest.raises(AssertionError):
        vi29.assert_clean(off + sub[1:], set(), {10, 11})            # chunk ngoai doclist
    with pytest.raises(AssertionError):
        vi29.assert_clean(sub[:1199], set(), {10, 11})


def test_require_sanity_1200_of_1200_else_exit2():
    vi29.require_sanity({"ok": True, "so_cau": 1200, "cau_khop_doc": 1200, "cau_khop_chunk": 1200})
    for bad in [{"ok": False, "so_cau": 1200, "cau_khop_doc": 1200, "cau_khop_chunk": 1200},
                {"ok": True, "so_cau": 1200, "cau_khop_doc": 1199, "cau_khop_chunk": 1200},
                {"ok": True, "so_cau": 1199, "cau_khop_doc": 1199, "cau_khop_chunk": 1199}]:
        with pytest.raises(SystemExit) as e:
            vi29.require_sanity(bad)
        assert e.value.code == 2


def fake_chain(max_delta_fit):
    def run(name, delta, max_zip, files, dedup_mode, validate=False):
        zb = 90_000_000 + 250_000 * delta
        fits = delta <= max_delta_fit
        return {"code": 0 if fits else 3,
                "next_delta": None if fits else max(max_delta_fit, 0),
                "zip_bytes": zb, "json_bytes": 1_000_000 * (100 + delta),
                "tong_chunk": 1200 * (100 + delta), "hist_kcap": [[100, 300, 1200]],
                "n_q_TB": 100.0}
    return run


def test_find_delta_lon_nhat_duoi_tran_zip_that(tmp_path, monkeypatch):
    monkeypatch.setattr(vi29, "OUT", tmp_path)
    monkeypatch.setattr(vi29, "chain_zip_check", fake_chain(max_delta_fit=52))
    d, attempts = vi29.find_delta({}, "doc_text", 99_000_000, 48, "probe")
    assert d == 52
    assert attempts[-1]["delta"] >= 52
    assert all(a["code"] == 0 for a in attempts if a["delta"] <= 52)


def test_find_delta_giam_khi_khoi_dau_vuot_tran(tmp_path, monkeypatch):
    monkeypatch.setattr(vi29, "OUT", tmp_path)
    monkeypatch.setattr(vi29, "chain_zip_check", fake_chain(max_delta_fit=30))
    d, _attempts = vi29.find_delta({}, "doc_text", 99_000_000, 60, "probe")
    assert d == 30


def test_find_delta_resume_theo_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vi29, "OUT", tmp_path)
    monkeypatch.setattr(vi29, "chain_zip_check", fake_chain(max_delta_fit=52))
    vi29.find_delta({}, "doc_text", 99_000_000, 52, "probe")
    calls = []

    def counted(name, delta, max_zip, files, dedup_mode, validate=False):
        calls.append(delta)
        return fake_chain(52)(name, delta, max_zip, files, dedup_mode, validate)

    monkeypatch.setattr(vi29, "chain_zip_check", counted)
    d, _ = vi29.find_delta({}, "doc_text", 99_000_000, 52, "probe")
    assert d == 52
    assert 52 not in calls


def test_loi_a_report_dem_dong_bi_loai_sai_va_topk():
    cands = {"1": [[99, 1, 0, 9.0, "x"], [10, 2, 0, 1.0, "x"]], "2": [[20, 1, 0, 1.0, "y"]]}
    doclists = {"1": ["10"], "2": ["20"]}
    kq = {"1": 1, "2": 1}
    rrf3d2 = {"1": [[99, 0, 1.0], [10, 1, 0.5]], "2": [[20, 0, 1.0]]}
    rep = vi29.loi_a_report(cands, doclists, kq, rrf3d2, wd=4.0, delta=57)
    assert rep["so_dong_bi_loai_sai"] == 1
    assert rep["so_cap_doc_text_bi_loai_sai"] == 1
    assert rep["so_cau_anh_huong"] == 1
    assert rep["so_cap_vao_topk_delta_57"] == 1
    assert rep["so_cau"] == 2
    assert rep["TB_C_q_moi"] == 1.0   # cau 1: (10,x) phuc hoi; cau 2: (20,y)
    assert rep["TB_C_q_cu"] == 0.5    # luat cu loai ca (10,x) vi 99 thang text


def test_sub_row_metrics():
    sub = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": [
            {"doc_id": 10, "chunk_text": "a"}, {"doc_id": 11, "chunk_text": "b"}]}]
    row = vi29.sub_row(sub, {"n_q_TB": 164.5}, {"bytes": 5, "sha256": "ab"}, 57)
    assert row["so_cau"] == 1 and row["delta"] == 57
    assert row["TB_doc_cau"] == 2.0 and row["TB_chunk_cau"] == 2.0
    assert row["TB_doc_co_chunk_cau"] == 2.0 and row["n_q_TB"] == 164.5
    assert row["zip_bytes"] == 5 and row["sha256"] == "ab"
    assert row["tong_chunk"] == 2


def test_percentile_va_dist():
    assert vi29.percentile([1, 2, 3, 4], 50) == 2.5
    assert vi29.percentile([1, 2, 3, 4], 90) == 3.7
    assert vi29.percentile([], 50) is None
    d = vi29.dist([4, 1, 3])
    assert d["n"] == 3 and d["min"] == 1 and d["max"] == 4 and d["TB"] == 2.7
    assert vi29.dist([]) == {"n": 0}


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class _StubTok:
    """Tokenizer do choi: moi tu = 1 token; cap (q, w) = sp2 + so tu; trunc cat xuong max."""

    def __init__(self, sp1=2, sp2=4, trunc=None):
        self.sp1, self.sp2, self.trunc = sp1, sp2, trunc

    def encode(self, q, w=None):
        if w is None:
            return _Enc([0] * (len(q.split()) + self.sp1))
        n = len(q.split()) + len(w.split()) + self.sp2
        return _Enc([0] * (min(n, self.trunc) if self.trunc is not None else n))


def test_token_meter_do_cat_theo_only_second():
    full = _StubTok()
    m = vi29.TokenMeter("tokenizers", full, _StubTok(trunc=512), 2, 4, {})
    q = " ".join(["q"] * 10)
    w = " ".join(["w"] * 600)
    nq, nw, pair, cut = m.measure(q, w)
    assert (nq, nw, pair) == (10, 600, 614)
    assert pair > vi29.MAX_LEN and cut == 614 - vi29.MAX_LEN
    m2 = vi29.TokenMeter("tokenizers", full, None, 2, 4, {})  # fallback cong thuc
    assert m2.measure(q, w) == (10, 600, 614, 102)
    assert m2.measure(q, " ".join(["w"] * 100))[3] == 0


def test_load_docs_wanted_giu_ban_truoc(tmp_path):
    import zstandard

    def write_jsonl_zst(p, rows):
        with p.open("wb") as f:
            with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
                for r in rows:
                    z.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))

    p1 = tmp_path / "a.jsonl.zst"
    p2 = tmp_path / "b.jsonl.zst"
    write_jsonl_zst(p1, [{"id": 1, "idx": 0, "text": "mot"}, {"id": 2, "idx": 1, "text": "hai"}])
    write_jsonl_zst(p2, [{"id": 1, "idx": 0, "text": "khac"}, {"id": 3, "idx": 1, "text": "ba"}])
    got = vi29.load_docs_wanted([p1, p2, tmp_path / "thieu.jsonl.zst"], {1, 2, 3, 4})
    assert got == {1: "mot", 2: "hai", 3: "ba"}


def test_export_missing_docs_tao_cache_va_goi_export(tmp_path, monkeypatch):
    import numpy as np
    ids_npy = tmp_path / "ids.npy"
    np.save(ids_npy, np.array([100, 200, 300]))
    monkeypatch.setattr(vi29, "IDX_IDS", ids_npy)
    monkeypatch.setattr(vi29, "TOK_DIR", tmp_path)
    calls = []

    def fake_run(script, *args):
        calls.append((script, [str(a) for a in args]))
        import zstandard
        out = tmp_path / "missing_docs.jsonl.zst"
        with out.open("wb") as f:
            with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
                z.write(json.dumps({"id": 200, "idx": 1, "text": "x"}, ensure_ascii=False).encode("utf-8"))

    monkeypatch.setattr(vi29, "run", fake_run)
    got = vi29.export_missing_docs({200, 999}, tmp_path / "missing_docs.jsonl.zst")
    assert got == {"yeu_cau": 2, "co_idx": 1, "ghi": 1,
                   "out": str(tmp_path / "missing_docs.jsonl.zst")}
    assert calls and calls[0][0] == "export_docs.py"
    cache = json.loads((tmp_path / "missing_cache.json").read_text(encoding="utf-8"))
    assert cache == {"tok": [[200, 1, 0.0]]}   # 999 khong co trong chi muc
