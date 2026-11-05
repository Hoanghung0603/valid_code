import json
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import runner_core as rc  # noqa: E402


# ---------------------------------------------------------------- truy van

def test_load_queries_parquet_and_jsonl(tmp_path):
    import pandas as pd
    pd.DataFrame({"id": [3, 1, 3, 2], "query": ["a", "b", "dup", ""]}).to_parquet(tmp_path / "q.parquet")
    assert rc.load_queries(tmp_path / "q.parquet") == [(3, "a"), (1, "b")]
    (tmp_path / "q.jsonl").write_text('{"id": 9, "query": "x"}\n{"id": 9, "query": "y"}\n'
                                      '{"id": 8, "zh": "z"}\n', encoding="utf-8")
    assert rc.load_queries(tmp_path / "q.jsonl") == [(9, "x"), (8, "z")]


def test_evenly_spaced_deterministic():
    ids = list(range(1, 1201))
    got = rc.evenly_spaced(ids, 20)
    assert len(got) == 20 and len(set(got)) == 20
    assert got == rc.evenly_spaced(list(reversed(ids)), 20)
    assert got[0] == 1 and got[-1] == 1200
    steps = [b - a for a, b in zip(got, got[1:])]
    assert max(steps) - min(steps) <= 1
    assert rc.evenly_spaced([5, 7], 5) == [5, 7]
    with pytest.raises(ValueError):
        rc.evenly_spaced(ids, 1)


# ---------------------------------------------------------------- pool / fuse

def test_pool_union_order_and_dedup():
    bm = [[1, 10], [2, 11], [3, 12]]
    dn = [[3, 12], [4, 13], [1, 10]]
    assert rc.pool_union(bm, dn, 200) == [[1, 10], [2, 11], [3, 12], [4, 13]]
    assert rc.pool_union(bm, dn, 2) == [[1, 10], [2, 11], [3, 12], [4, 13]]  # moi nguon lay top-2


def test_sort_by_score_none_last_stable():
    hits = [[1, 0, 0.5], [2, 1, None], [3, 2, 0.5], [4, 3, 0.9]]
    assert [h[0] for h in rc.sort_by_score(hits)] == [4, 1, 3, 2]


def test_fuse_rrf_weights():
    pool = {"7": [[10, 0, 1.0], [11, 1, 0.5], [12, 2, 0.0]]}
    dense = {"7": [[10, 0, 0.9], [11, 1, 0.8]]}
    q8b = {"7": [[12, 2, 0.3]]}
    hyde = {"7": [[12, 2, 0.6]]}
    rrf3d2, fhyde = rc.fuse_rrf(pool, q8b, dense, hyde, w8f=2.0)

    def rank(hits):
        return {str(h[0]): i + 1 for i, h in enumerate(hits)}

    rb, rd, r8, rh = rank(pool["7"]), rank(dense["7"]), rank(q8b["7"]), rank(hyde["7"])
    for out, w8b, why in ((rrf3d2, 1.0, 0.0), (fhyde, 2.0, 1.0)):
        exp = {}
        for d in ("10", "11", "12"):
            sc = 0.0
            for w, r in ((1.0, rb.get(d)), (w8b, r8.get(d)), (2.0, rd.get(d)), (why, rh.get(d))):
                if w and r:
                    sc += w / (60 + r)
            exp[d] = sc
        got = {str(h[0]): h[2] for h in out["7"]}
        for d in exp:
            assert got[d] == pytest.approx(round(exp[d], 8), abs=1e-8)
        assert [str(h[0]) for h in out["7"]] == sorted(exp, key=lambda d: -exp[d])
    assert {str(h[0]) for h in fhyde["7"]} == {"10", "11", "12"}


def test_n_q_clamp_and_threshold():
    hits = [[1, 0, -1.0], [2, 1, -2.5], [3, 2, 0.0], [4, 3, -2.0]]
    assert rc.n_q_for(hits, -2.0, 1, 400) == 2
    assert rc.n_q_for(hits, -2.0, 5, 400) == 5
    assert rc.n_q_for(hits, -2.0, 1, 2) == 2
    assert rc.n_q_for([[1, 0, None]], -2.0, 4, 400) == 4


def test_doclist_first_n_and_top_bge():
    fhyde = [[3, 0, 1.0], [3, 0, 1.0], [5, 1, 0.5], [9, 2, 0.1]]
    assert rc.doclist_for(fhyde, 2) == [3, 5]
    pool = [[8, 0, 2.0], [7, 1, 1.0], [6, 2, 0.0]]
    assert rc.top_bge(pool, 2) == [8, 7]


# ---------------------------------------------------------------- 8B

def test_q8b_pairs_doclist_plus_margin():
    fhyde = {"1": [[d, 0, 1.0 - d / 100] for d in range(1, 11)],
             "2": [[d, 0, 1.0] for d in (20, 21)]}
    pool = {"1": [[d, 0, 0] for d in range(1, 11)], "2": [[20, 0, 0], [21, 0, 0]]}
    dl = {"1": [1, 2, 3], "2": [20]}
    got = rc.q8b_pairs_for(fhyde, pool, dl, scored={}, margin=2)
    assert got["1"] == [1, 2, 3, 4, 5]    # top-(3+2) cua fhyde, ke ca doc trong doclist
    got2 = rc.q8b_pairs_for(fhyde, pool, dl, scored={"1": [[4, 0, 0.1]]}, margin=2)
    assert got2["1"] == [1, 2, 3, 5]
    assert got2["2"] == [20, 21]


def test_q8b_pairs_skips_outside_pool_and_rescored_none():
    fhyde = {"1": [[1, 0, 1.0], [2, 0, 0.9], [3, 0, 0.8]]}
    pool = {"1": [[1, 0, 0], [3, 0, 0]]}
    got = rc.q8b_pairs_for(fhyde, pool, {"1": [1]}, scored={"1": [[1, 0, 0.2], [3, 0, None]]},
                           margin=2)
    assert got["1"] == [3]  # 2 ngoai pool; 1 da co diem; 3 diem None -> phai cham lai


def test_merge_q8b_drop_none_keep_old_sort():
    old = {"1": [[1, 0, 0.5]]}
    new = {"1": [[2, 1, 0.9], [1, 0, 0.7], [3, 2, None]]}
    merged, st = rc.merge_q8b(old, new)
    assert merged["1"] == [[2, 1, 0.9], [1, 0, 0.5]]
    assert st == {"trung_bo": 1, "none_bo": 1}


# ---------------------------------------------------------------- k_q noi tai

def _rows(pairs):
    return [[d, i + 1, i, lg, t] for i, (d, lg, t) in enumerate(pairs)]


def test_intrinsic_kq_cap_then_filter_vs_filter_then_cap():
    rows = _rows([(10, 5.0, "a"), (11, 4.0, "b"), (12, 3.0, "c"), (13, 2.0, "d")])
    fused = [[10, 0, 1.0], [11, 1, 1.0], [12, 2, 1.0], [13, 3, 1.0]]  # zd deu nhau -> thu tu = logit
    allowed = {11, 12, 13}
    assert rc.intrinsic_kq(rows, fused, allowed, cap=2, mode=0) == 1   # cat 2 truoc: [10,11] -> 11
    assert rc.intrinsic_kq(rows, fused, allowed, cap=2, mode=1) == 2   # loc truoc: [11,12] -> 2
    assert rc.intrinsic_kq(rows, fused, allowed, cap=99, mode=0) == 3
    assert rc.intrinsic_kq([], [], allowed) == 0


def test_intrinsic_kq_dedup_modes():
    rows = _rows([(10, 5.0, "same"), (11, 4.0, "same"), (12, 3.0, "x")])
    fused = [[10, 0, 1.0], [11, 1, 1.0], [12, 2, 1.0]]
    allowed = {10, 11, 12}
    assert rc.intrinsic_kq(rows, fused, allowed, cap=99, dedup_mode="doc_text") == 3
    assert rc.intrinsic_kq(rows, fused, allowed, cap=99, dedup_mode="text") == 2


# ---------------------------------------------------------------- chain

def _formula_sub(pool, q8b, dense, hyde, cands, kq, delta, lo, hi, wd=4.0, b_cov=1.0):
    """Tai lap doc lap cong thuc vi34 (RRF + zscore + s = zc + wd*zd + b_cov + k') de doi chieu."""
    from chunk_docfuse import zscores

    def ranks(hits):
        return {str(h[0]): i + 1 for i, h in enumerate(hits)}

    sub = []
    for q in sorted(pool, key=lambda x: int(x)):
        rb, rd = ranks(pool[q]), ranks(dense.get(q, []))
        r8, rh = ranks(q8b.get(q, [])), ranks(hyde.get(q, []))
        fused = {}
        for h in pool[q]:
            d = str(h[0])
            fused[d] = sum(w / (60 + r) for w, r in ((1.0, rb.get(d)), (1.0, r8.get(d)),
                                                     (2.0, rd.get(d)), (0.0, rh.get(d)))
                           if w and r)
        fh = {}
        for h in pool[q]:
            d = str(h[0])
            fh[d] = sum(w / (60 + r) for w, r in ((1.0, rb.get(d)), (2.0, r8.get(d)),
                                                  (2.0, rd.get(d)), (1.0, rh.get(d)))
                        if w and r)
        n = min(max(sum(1 for h in pool[q] if h[2] is not None and h[2] > -2.0), lo), hi)
        order = sorted(fh, key=lambda d: -fh[d])[:n]
        zdm = dict(zip(map(str, (h[0] for h in pool[q])),
                       zscores([fused[str(h[0])] for h in pool[q]])))
        zmin = min(zdm.values()) if zdm else 0.0
        rows = cands.get(q, [])
        zc = zscores([float(r[3]) for r in rows])
        scored = [(c + wd * zdm.get(str(r[0]), zmin), c, r) for c, r in zip(zc, rows)]
        srt = sorted(scored, key=lambda t: -t[0])
        seen, picked = set(), []
        allowed = {str(d) for d in order}
        for s, c, r in srt:
            if str(r[0]) not in allowed or (str(r[0]), r[4]) in seen:
                continue
            seen.add((str(r[0]), r[4]))
            picked.append({"doc_id": r[0], "chunk_text": r[4], "s": s})
        keys, seen2 = [], set()
        for c in picked:
            d = str(c["doc_id"])
            keys.append(c["s"] + (b_cov if d not in seen2 else 0.0))
            seen2.add(d)
        picked = [picked[i] for i in sorted(range(len(picked)), key=lambda i: -keys[i])]
        kp = min(kq.get(q, 0) + delta, len(picked))
        sub.append({"id": int(q), "relevant_docs": [int(d) for d in order],
                    "relevant_chunks": [{"doc_id": c["doc_id"], "chunk_text": c["chunk_text"]}
                                        for c in picked[:kp]]})
    return sub


def _chain_data():
    pool = {"7": [[10, 0, 1.0], [11, 1, 0.5], [12, 2, 0.0]],
            "42": [[20, 3, 0.9], [21, 4, -3.0]]}
    q8b = {"7": [[11, 1, 0.3]]}
    dense = {"7": [[10, 0, 0.9], [11, 1, 0.8]], "42": [[20, 3, 0.7]]}
    hyde = {"7": [[12, 2, 0.6]]}
    cands = {"7": _rows([(10, 1.0, "a"), (10, 0.0, "b"), (11, 0.5, "c"), (12, -1.0, "d")]),
             "42": _rows([(20, 0.7, "e"), (21, -2.0, "f")])}
    kq = {"7": 1, "42": 0}
    return pool, q8b, dense, hyde, cands, kq


def test_chain_build_runner_arbitrary_ids_and_formula():
    pool, q8b, dense, hyde, cands, kq = _chain_data()
    rrf3d2, fhyde, sub, rep = rc.chain_build_runner(pool, q8b, dense, hyde, cands, kq, delta=1,
                                                    lo=1, hi=2)
    assert [it["id"] for it in sub] == [7, 42]
    want = _formula_sub(pool, q8b, dense, hyde, cands, kq, delta=1, lo=1, hi=2)
    for got, exp in zip(sub, want):
        assert got["relevant_docs"] == exp["relevant_docs"]
        assert [(c["doc_id"], c["chunk_text"]) for c in got["relevant_chunks"]] == \
               [(c["doc_id"], c["chunk_text"]) for c in exp["relevant_chunks"]]
    assert rep["delta"] == 1 and rep["dedup_mode"] == "doc_text"
    assert rep["TB_chunk_cau"] == pytest.approx(
        sum(len(it["relevant_chunks"]) for it in sub) / 2)


def test_chain_no_ref_chunks_and_kprime_cap():
    pool, q8b, dense, hyde, cands, kq = _chain_data()
    _, _, sub0, _ = rc.chain_build_runner(pool, q8b, dense, hyde, cands, {"7": 0, "42": 0},
                                          delta=0, lo=1, hi=2)
    assert all(len(it["relevant_chunks"]) == 0 for it in sub0)   # k' = min(0+0, |C_q|) = 0
    _, _, sub2, _ = rc.chain_build_runner(pool, q8b, dense, hyde, cands, {"7": 99, "42": 99},
                                          delta=50, lo=1, hi=2)
    exp = _formula_sub(pool, q8b, dense, hyde, cands, {"7": 99, "42": 99}, delta=50, lo=1, hi=2)
    for it, e in zip(sub2, exp):
        assert len(it["relevant_chunks"]) == len(e["relevant_chunks"])   # k' bi chan boi |C_q|
        assert {str(c["doc_id"]) for c in it["relevant_chunks"]} <= \
               {str(d) for d in it["relevant_docs"]}
        assert set(it["relevant_docs"]) <= {10, 11, 12, 20, 21}          # khong them doc ref


def test_run_chain_runner_writes_files(tmp_path):
    pool, q8b, dense, hyde, cands, kq = _chain_data()
    rep = rc.run_chain_runner(tmp_path, "t", 1, pool, q8b, dense, hyde, cands, kq, lo=1, hi=2)
    assert (tmp_path / "submission_t.json").exists()
    assert (tmp_path / "report_chain_t.json").exists()
    sub = json.loads((tmp_path / "submission_t.json").read_text(encoding="utf-8"))
    assert [it["id"] for it in sub] == [7, 42]
    assert rep["json_bytes"] == (tmp_path / "submission_t.json").stat().st_size


# ---------------------------------------------------------------- assert_clean / so sanh

def _sub_ok():
    return [{"id": 999999, "relevant_docs": [5, 6],
             "relevant_chunks": [{"doc_id": 5, "chunk_text": "a"}]},
            {"id": 7, "relevant_docs": [8],
             "relevant_chunks": [{"doc_id": 8, "chunk_text": "b"}]}]


def test_assert_clean_runner_arbitrary_ids():
    clean = rc.assert_clean_runner(_sub_ok(), bad=[1, 2], corpus={5, 6, 8}, expect_ids=[7, 999999])
    assert clean["cau"] == 2 and clean["cap_doc_text_trung"] == 0


def test_assert_clean_runner_failures():
    with pytest.raises(AssertionError):
        rc.assert_clean_runner(_sub_ok(), bad=[5], corpus={5, 6, 8})          # bad trong docs
    with pytest.raises(AssertionError):
        rc.assert_clean_runner(_sub_ok(), bad=[], corpus={6, 8})              # doc ngoai kho
    sub = _sub_ok()
    sub[0]["relevant_chunks"].append({"doc_id": 5, "chunk_text": "a"})       # trung (doc_id, text)
    with pytest.raises(AssertionError):
        rc.assert_clean_runner(sub, bad=[], corpus={5, 6, 8})
    sub2 = _sub_ok()
    sub2[0]["relevant_chunks"].append({"doc_id": 6, "chunk_text": "a"})       # cung text, khac doc: OK
    assert rc.assert_clean_runner(sub2, bad=[], corpus={5, 6, 8})["cap_doc_text_trung"] == 0
    sub3 = _sub_ok()
    sub3[0]["relevant_chunks"] = [{"doc_id": 6, "chunk_text": "z"}]
    sub3[0]["relevant_docs"] = [5]
    with pytest.raises(AssertionError):
        rc.assert_clean_runner(sub3, bad=[], corpus={5, 6, 8})               # chunk ngoai doclist
    with pytest.raises(AssertionError):
        rc.assert_clean_runner(_sub_ok(), bad=[], corpus={5, 6, 8}, expect_ids=[7])  # thieu id


def test_compare_subs_subset():
    got = _sub_ok()
    ref = [{"id": 999999, "relevant_docs": [5, 9], "relevant_chunks": [{"doc_id": 5, "chunk_text": "a"}]},
           {"id": 7, "relevant_docs": [], "relevant_chunks": []}]
    rep = rc.compare_subs_subset(got, ref, top=2)
    assert rep["so_cau"] == 2
    assert rep["TB_top2_doc_trung"] == pytest.approx(0.5)
    assert rep["TB_chunk_trung_nguyen_van"] == pytest.approx(0.5)
    assert rep["TB_chunk_moi_cau"] == pytest.approx(0.5)


# ---------------------------------------------------------------- delta

def _fake_attempt_factory(kq_cap_pairs, unit_json=8000.0, ratio=79_380_644 / 505_693_628,
                          limit=200_000):
    from cx110 import hist_from_pairs, suggest_next_delta, total_chunks
    hist = hist_from_pairs(kq_cap_pairs)
    tried = []

    def attempt(d):
        tried.append(d)
        n = total_chunks(hist, d)
        jb = unit_json * n
        zb = jb * ratio
        code = 0 if zb <= limit else 3
        return {"code": code,
                "next_delta": suggest_next_delta(hist, d, n, zb, limit, zb / max(n, 1)) if code == 3 else None,
                "zip_bytes": zb, "json_bytes": jb, "tong_chunk": n, "hist_kcap": hist}
    return attempt, tried


LIMIT = 200_000


def test_find_delta_runner_largest_fitting():
    from cx110 import hist_from_pairs, total_chunks
    pairs = [(50, 200), (80, 300), (10, 60)]
    hist = hist_from_pairs(pairs)
    attempt, tried = _fake_attempt_factory(pairs, limit=LIMIT)
    delta, attempts = rc.find_delta_runner(attempt, delta0=200, max_zip=LIMIT)
    ok = [d for d in range(0, 400)
          if 8000.0 * total_chunks(hist, d) * (79_380_644 / 505_693_628) <= LIMIT]
    assert delta == max(ok)
    assert len(tried) == len(set(tried)) <= 16
    assert all(a["code"] in (0, 3) for a in attempts)


def test_find_delta_runner_saturated_capacity():
    # |C_q| = 1 moi cau: tang delta khong them chunk nua -> dung o delta0 (vi29.find_delta)
    pairs = [(0, 1)] * 5
    attempt, _ = _fake_attempt_factory(pairs, unit_json=1000.0, limit=10 ** 12)
    delta, _ = rc.find_delta_runner(attempt, delta0=10, max_zip=10 ** 12)
    assert delta == 10


def test_find_delta_runner_descends_with_next_delta():
    pairs = [(200, 400)]
    attempt, _ = _fake_attempt_factory(pairs, unit_json=100.0, limit=5_000)
    delta, _ = rc.find_delta_runner(attempt, delta0=300, max_zip=5_000)
    assert 0 < delta < 300


# ---------------------------------------------------------------- bench/manifest

def test_sample_cache_pairs_and_corr():
    cache = {"1": [[1, 0, 0.5], [2, 1, None]], "2": [[3, 0, 0.1]]}
    s = rc.sample_cache_pairs(cache, 2, seed=1)
    assert len(s) == 2 and all(x[2] is not None for x in s)
    wins = {"1": {"9": [[0, 10, 0.5], [5, 15, 0.2]]}}
    w = rc.sample_docwin_rows(wins, 5, seed=1)
    assert len(w) == 2 and w[0][0] == "1"
    r = rc.corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r["pearson"] == pytest.approx(1.0) and r["lech_max"] == 0.0


def test_validate_runner_reports_failure(tmp_path):
    import runner_private as rp
    bad = tmp_path / "bad.json"
    bad.write_text('[{"id": 1, "relevant_docs": "khong phai list"}]', encoding="utf-8")
    r = rp.validate_runner(bad, 1)
    assert r["code"] != 0 and r["output"]
    r2 = rp.validate_runner(tmp_path / "khong_ton_tai.json", 1)
    assert r2["code"] != 0


def test_pool_for_mode_base_is_current_definition_and_subset_of_p1():
    bm = [[i, 100 + i] for i in range(1, 301)]          # BM25 top-300 (cache k300)
    dn = [[i, 200 + i] for i in range(250, 450)]        # dense top-200
    hy = [[i, 300 + i] for i in range(400, 600)]        # HyDE top-200
    base = rc.pool_for_mode("base", bm, dn, hy)
    assert base == rc.pool_union(bm[:200], dn, 200)     # dinh nghia hien tai (BM25 top-200)
    p1 = rc.pool_for_mode("p1", bm, dn, hy)
    assert {str(h[0]) for h in base} <= {str(h[0]) for h in p1}
    assert [h[0] for h in p1[:200]] == [h[0] for h in base[:200]]   # BM25 top-200 la prefix chung
    assert len({str(h[0]) for h in p1}) == len(p1)
    with pytest.raises(ValueError):
        rc.pool_for_mode("xxx", bm, dn, hy)


def test_pool_for_mode_base_reproduces_rsmoke20_pool_shape():
    # base phai tai lap quy tac pool_union(bm25[:200], dense[:200]) dung cho pool_cands rsmoke20
    bm = [[i, 1] for i in range(1, 301)]
    dn = [[i, 1] for i in range(1000, 1200)]
    assert len(rc.pool_for_mode("base", bm, dn, [])) == 400


def test_partwriter_rotates_by_bytes(tmp_path):
    import os
    import random
    import runner_private as rp
    import secrets
    w = rp.PartWriter(tmp_path / "docs_t", part_mb=1, log=lambda m: None)
    for i in range(8000):   # hex ngau nhien ~ 4 bit/ky tu -> nen kem, du de vuot 1 MB
        w.write({"id": i, "idx": i, "text": secrets.token_hex(200)})
    w.close()
    assert len(w.paths) >= 2
    sizes = [os.path.getsize(p) for p in w.paths]
    assert all(s < 1_200_000 for s in sizes[:-1]), sizes  # part day ~ 1 MB nen (khong theo so ban ghi)
    assert w.n == 8000


def test_stats_kq_blocked_and_doclist_overlap():
    import runner_private as rp
    sub = [{"id": 1, "relevant_docs": [5, 6], "relevant_chunks": [{"doc_id": 5, "chunk_text": "a"}]},
           {"id": 2, "relevant_docs": [7], "relevant_chunks": [{"doc_id": 7, "chunk_text": "b"}] * 3}]
    kq = {"1": 1, "2": 2}
    # cau 1: kq+delta = 3 > 1 chunk -> bi chan; cau 2: kq+delta = 4 > 3 chunk -> bi chan
    assert rp.kq_blocked(sub, kq, delta=2) == 2
    assert rp.kq_blocked(sub, kq, delta=0) == 0   # kq=1 vs 1 chunk, kq=2 vs 3 chunk: khong chan
    ref = [{"id": 1, "relevant_docs": [5, 9]}, {"id": 2, "relevant_docs": [8]}]
    assert rp.doclist_cap_ref(sub, ref) == pytest.approx(0.5)   # cau 1 giao {5}, cau 2 giao rong


def test_write_manifest_sha(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    man = rc.write_manifest(tmp_path, "s1", {"k": 1}, [tmp_path / "a.txt"], [tmp_path / "missing.json"],
                            1.5, machine="b660")
    d = json.loads(man.read_text(encoding="utf-8"))
    assert d["stage"] == "s1" and d["inputs"][0]["sha256"]
    assert d["outputs"][0]["missing"] and d["machine"] == "b660"
