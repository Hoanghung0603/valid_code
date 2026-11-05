import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dwx_cands import band_of, build, idx_lookup, load_json, q_d_docs, top_ids  # noqa: E402


def _fixtures():
    pool = {
        "1": [[1000 + i, 3000 + i, 5.0 - i * 0.01] for i in range(100)],
        "2": [[2000 + i, 4000 + i, 5.0 - i * 0.01] for i in range(100)],
    }
    fhyde = {
        "1": [[50, 5000, 0.9], [99, 9900, 0.8]],
        "2": [[22, 4022, 0.7], [300, 5300, 0.6]],
    }
    split = [
        {"id": 1, "relevant_docs": [50, 1005, 99, 7, 8], "relevant_chunks": [{"doc_id": 50, "chunk_text": "x"}]},
        {"id": 2, "relevant_docs": [2005, 22, 300], "relevant_chunks": []},
    ]
    return split, pool, fhyde


def test_top_ids_first_n():
    hits = [[1, 0, 1.0], [2, 0, 0.9], [3, 0, 0.8]]
    assert top_ids(hits, 2) == {"1", "2"}
    assert top_ids(None, 2) == set()


def test_idx_lookup_pool_preferred_fhyde_fallback():
    pool_hits = [[10, 1000, 5.0]]
    fhyde_hits = [[10, 9999, 0.9], [50, 5000, 0.8], [60, None, 0.7]]
    assert idx_lookup(pool_hits, fhyde_hits) == {"10": 1000, "50": 5000}


def test_q_d_docs_excludes_top100_keeps_rank_and_flags_missing():
    split, pool, fhyde = _fixtures()
    t100 = top_ids(pool["1"], 100)
    idxs = idx_lookup(pool["1"], fhyde.get("1", []))
    dq, missing = q_d_docs(split[0]["relevant_docs"], t100, idxs)
    assert dq == [(1, 50, True), (3, 99, True), (4, 7, False), (5, 8, False)]
    assert missing == [[4, 7], [5, 8]]


def test_q_d_docs_missing_idx_reported():
    dq, missing = q_d_docs([5, 6], {"9"}, {})
    assert dq == [(1, 5, False), (2, 6, False)]
    assert missing == [[1, 5], [2, 6]]


def test_q_d_docs_rank_band_filters_but_ranks_unchanged():
    split, pool, fhyde = _fixtures()
    t100 = top_ids(pool["1"], 100)
    idxs = idx_lookup(pool["1"], fhyde.get("1", []))
    dq, missing = q_d_docs(split[0]["relevant_docs"], t100, idxs, rank_band=(3, 4))
    assert [r for r, _, _ in dq] == [3, 4]
    assert [d for _, d, _ in dq] == [99, 7]
    assert missing == [[4, 7]]


def test_build_counts_and_outputs():
    split, pool, fhyde = _fixtures()
    cands_full, cache_ext, rep = build(split, pool, fhyde, top_n=100)
    assert list(cands_full.keys()) == ["1", "2"]
    assert cands_full["1"] == [[50, 5000], [1005, 3005], [99, 9900], [7, None], [8, None]]
    assert cands_full["2"] == [[2005, 4005], [22, 4022], [300, 5300]]
    assert cache_ext["1"] == [[50, 5000, 0.0], [99, 9900, 0.0]]
    assert cache_ext["2"] == [[22, 4022, 0.0], [300, 5300, 0.0]]
    assert rep["cặp (q, doc) D_q"] == 6
    assert rep["cặp export được (có idx)"] == 4
    assert rep["TB D_q/câu"] == 3.0
    assert rep["D_q min/max"] == [2, 4]
    assert rep["doc duy nhất"] == 6
    assert rep["cặp theo dải hạng nộp"]["1-30"] == 6
    assert rep["doc thiếu idx"] == [["1", 4, 7], ["1", 5, 8]]
    assert rep["rank_band"] is None


def test_build_rank_band_only_exports_band():
    split, pool, fhyde = _fixtures()
    cands_full, cache_ext, rep = build(split, pool, fhyde, top_n=100, rank_band=(3, 4))
    assert cache_ext["1"] == [[99, 9900, 0.0]]
    assert cache_ext["2"] == [[300, 5300, 0.0]]
    assert len(cands_full["1"]) == 5
    assert rep["cặp (q, doc) D_q"] == 3
    assert rep["rank_band"] == [3, 4]


def test_build_string_ids():
    pool = {"7": [[10, 1, 1.0]]}
    fhyde = {"7": [["88", 2, 0.5]]}
    split = [{"id": "7", "relevant_docs": ["88", 10], "relevant_chunks": []}]
    cands_full, cache_ext, rep = build(split, pool, fhyde, top_n=100)
    assert cache_ext["7"] == [[88, 2, 0.0]]
    assert cands_full["7"] == [[88, 2], [10, 1]]
    assert rep["cặp (q, doc) D_q"] == 1
    assert rep["doc duy nhất"] == 1


def test_band_of():
    assert band_of(1) == "1-30"
    assert band_of(30) == "1-30"
    assert band_of(31) == "31-60"
    assert band_of(61) == "61-100"
    assert band_of(101) == "101+"
    assert band_of(500) == "101+"


def test_load_json_plain_and_zst(tmp_path):
    import zstandard
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert load_json(str(p)) == {"x": 1}
    z = tmp_path / "b.json.zst"
    with open(z, "wb") as f:
        f.write(zstandard.ZstdCompressor().compress(json.dumps([1, 2]).encode("utf-8")))
    assert load_json(str(z)) == [1, 2]
