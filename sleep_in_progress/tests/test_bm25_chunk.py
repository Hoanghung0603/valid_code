import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pytest

import bm25_chunk
from bm25_chunk import (
    IdSeq,
    aggregate_hits,
    bench_tokenizer,
    chunk_spans,
    count_docs,
    expand_docs,
    load_queries,
    make_tokenizer,
    overlap_mean,
    read_json_zst,
    report,
    run,
    sha256_file,
    syllable_spans,
    tokenize_pyvi,
    tokenize_syl,
    top_doc_ids,
    write_json_zst,
)


def test_syllable_spans_offsets():
    assert syllable_spans("x  y z") == [(0, 1), (3, 4), (5, 6)]
    assert syllable_spans("   ") == []
    assert syllable_spans("") == []


def test_chunk_short_doc_single_chunk_verbatim():
    t = "sốt xuất huyết"
    (idx, s, e), = chunk_spans(t)
    assert (idx, s, e) == (0, 0, len(t))
    assert t[s:e] == t
    t2 = "  ab cd  "
    (idx, s, e), = chunk_spans(t2)
    assert (idx, t2[s:e]) == (0, "ab cd")


def test_chunk_two_windows_512_50_overlap():
    toks = [f"w{i}" for i in range(600)]
    text = " ".join(toks)
    sp = chunk_spans(text)
    assert [c for c, _, _ in sp] == [0, 1]
    c0 = text[sp[0][1]:sp[0][2]]
    c1 = text[sp[1][1]:sp[1][2]]
    assert c0 == " ".join(toks[:512])
    assert c1 == " ".join(toks[462:])
    assert c0 in text and c1 in text
    assert c0.split()[462:] == c1.split()[:50]


def test_chunk_window_boundaries():
    for n, expect in [(512, 1), (513, 2), (924, 2), (925, 2), (974, 2), (975, 3)]:
        text = " ".join(f"w{i}" for i in range(n))
        assert len(chunk_spans(text)) == expect, n


def test_chunk_empty_text_one_empty_chunk():
    assert chunk_spans("") == [(0, 0, 0)]
    assert chunk_spans("  \n\t ") == [(0, 0, 0)]


def test_chunk_custom_win_overlap_and_bad_params():
    text = " ".join(str(i) for i in range(10))
    sp = chunk_spans(text, win=4, overlap=1)
    assert [text[s:e] for _, s, e in sp] == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    with pytest.raises(ValueError):
        chunk_spans(text, win=4, overlap=4)
    with pytest.raises(ValueError):
        chunk_spans(text, win=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_spans(text, win=4, overlap=-1)


def test_tokenize_syl_nfkc_case_nonword():
    assert tokenize_syl("ＡＢＣ Sốt_Xuất") == ["abc", "sốt", "xuất"]
    assert tokenize_syl("Đau, Dạ-Dày!") == ["đau", "dạ", "dày"]


def test_tokenize_syl_cjk_single_tokens():
    assert tokenize_syl("五倍子膏 B12") == ["五", "倍", "子", "膏", "b12"]


def test_tokenize_syl_keeps_diacritics_no_bigrams():
    assert tokenize_syl("sốt xuất huyết") == ["sốt", "xuất", "huyết"]


def test_tokenize_pyvi_fake(monkeypatch):
    class Fake:
        @staticmethod
        def tokenize(text):
            return "Xin chào_bạn ."

    monkeypatch.setattr(bm25_chunk, "_pyvi", Fake)
    assert tokenize_pyvi("whatever") == ["xin", "chào_bạn", "."]


def test_tokenize_pyvi_real():
    pytest.importorskip("pyvi")
    toks = tokenize_pyvi("Tôi bị đau dạ dày.")
    assert "đau_dạ_dày" in toks
    assert all(t == t.lower() for t in toks)


def test_make_tokenizer():
    assert make_tokenizer("syl") is tokenize_syl
    assert make_tokenizer("pyvi") is tokenize_pyvi
    with pytest.raises(ValueError):
        make_tokenizer("xx")


def test_aggregate_hits_max_chunk_score_and_order():
    meta = [(10, 0, 0, 5), (10, 1, 3, 8), (20, 0, 0, 9), (30, 0, 0, 2)]
    hits = [(1, 2.0), (0, 1.5), (2, 2.0), (3, 0.5)]
    out = aggregate_hits(hits, meta, top_docs=2)
    assert out == [(10, 2.0, 1, 3, 8), (20, 2.0, 0, 0, 9)]
    out3 = aggregate_hits(hits, meta, top_docs=3)
    assert out3[2] == (30, 0.5, 0, 0, 2)


def test_aggregate_hits_tie_keeps_first_chunk():
    meta = [(10, 0, 0, 5), (10, 1, 3, 8)]
    assert aggregate_hits([(0, 2.0), (1, 2.0)], meta, top_docs=5) == [(10, 2.0, 0, 0, 5)]


def test_top_doc_ids_and_overlap_mean():
    cache = {"1": [[5, 0, 1.0], [6, 1, 0.9], [7, 2, 0.8]], "2": [[1, 0, 1.0]]}
    ref = {"1": [[7, 0, 1.0], [6, 1, 0.9], [8, 2, 0.8]], "2": [[2, 0, 1.0]], "3": [[1, 0, 1.0]]}
    assert top_doc_ids(cache["1"], 2) == [5, 6]
    assert overlap_mean(cache, ref, 2) == pytest.approx(0.5)
    assert overlap_mean({}, ref, 30) == 0.0


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_expand_docs_glob_and_missing(tmp_path):
    for n in ["b.jsonl.zst", "a.jsonl.zst"]:
        (tmp_path / n).write_bytes(b"x")
    got = expand_docs([str(tmp_path / "*.jsonl.zst")])
    assert [Path(g).name for g in got] == ["a.jsonl.zst", "b.jsonl.zst"]
    assert expand_docs([str(tmp_path / "a.jsonl.zst")]) == [str(tmp_path / "a.jsonl.zst")]
    with pytest.raises(FileNotFoundError):
        expand_docs([str(tmp_path / "missing.jsonl.zst")])
    with pytest.raises(FileNotFoundError):
        expand_docs([str(tmp_path / "*.nomatch")])


def test_load_queries_jsonl_and_parquet(tmp_path):
    jl = tmp_path / "q.jsonl"
    jl.write_text('{"id": 7, "query": "đau bụng"}\n{"id": 8, "query": "sốt"}\n', encoding="utf-8")
    assert load_queries(jl) == [(7, "đau bụng"), (8, "sốt")]
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pq = tmp_path / "q.parquet"
    pd.DataFrame({"id": [1, 2], "query": ["a", "b"]}).to_parquet(pq)
    assert load_queries(pq) == [(1, "a"), (2, "b")]


def test_write_read_json_zst_roundtrip(tmp_path):
    pytest.importorskip("zstandard")
    p = tmp_path / "x.json.zst"
    obj = {"1": [[3, 0, 1.5, "sốt xuất"]]}
    write_json_zst(obj, p)
    assert read_json_zst(p) == obj


def test_count_docs_and_bench_tokenizer(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    p = tmp_path / "d.jsonl.zst"
    recs = [{"id": 1, "text": "một hai"}, {"id": 2, "text": "ba bốn"}, {"id": 3, "text": "năm"}]
    with open(p, "wb") as fh:
        fh.write(zstandard.ZstdCompressor().compress(
            ("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n").encode("utf-8")))
    assert count_docs([str(p)]) == 3
    seen, chars, el = bench_tokenizer([str(p)], 2, tokenize_syl)
    assert (seen, chars) == (2, len("một hai") + len("ba bốn"))
    assert el >= 0


def test_peak_ram_mb():
    ram = bm25_chunk.peak_ram_mb()
    assert ram is None or (len(ram) == 2 and all(x > 0 for x in ram))


def test_idseq_iteration():
    flat = np.array([5, 6, 7, 8], dtype=np.int32)
    seq = IdSeq(flat, [0, 2, 3, 4])
    assert len(seq) == 3
    assert seq[0] == [5, 6] and seq[2] == [8]
    assert list(seq) == [[5, 6], [7], [8]]


def _write_docs(path, recs):
    import zstandard
    with open(path, "wb") as fh:
        fh.write(zstandard.ZstdCompressor().compress(
            ("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n").encode("utf-8")))


def test_run_end_to_end(tmp_path):
    pytest.importorskip("bm25s")
    pytest.importorskip("zstandard")
    docs = [
        {"id": 101, "text": "sốt xuất huyết dengue muỗi vằn sốt cao"},
        {"id": 102, "text": "đau dạ dày viêm loét hành tá tràng"},
        {"id": 103, "text": "tiểu đường tuýp 2 insulin huyết áp"},
    ]
    dp = tmp_path / "docs.jsonl.zst"
    _write_docs(dp, docs)
    qp = tmp_path / "q.jsonl"
    qp.write_text('{"id": 1, "query": "muỗi vằn gây sốt xuất huyết"}\n'
                  '{"id": 2, "query": "viêm loét dạ dày"}\n'
                  '{"id": 3, "query": ""}\n', encoding="utf-8")
    out = tmp_path / "out"
    stats = run([str(dp)], str(qp), "syl", out, top_chunks=10, top_docs=2)
    assert stats and stats["docs"] == 3 and stats["chunks"] == 3
    cache = json.loads((out / "cache_bm25c_syl.json").read_text(encoding="utf-8"))
    assert set(cache) == {"1", "2", "3"}
    assert cache["3"] == []
    for q in ["1", "2"]:
        hits = cache[q]
        assert [h[1] for h in hits] == list(range(len(hits)))
        scores = [h[2] for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(isinstance(h[0], int) for h in hits)
    assert cache["1"][0][0] == 101
    assert cache["2"][0][0] == 102
    best = read_json_zst(out / "bestchunk_bm25c_syl.json.zst")
    by_id = {d["id"]: d["text"] for d in docs}
    assert [r[0] for r in best["1"]] == [h[0] for h in cache["1"]]
    for row in best["1"] + best["2"]:
        doc_id, cidx, sc, text = row
        assert cidx == 0 and isinstance(sc, float)
        assert text in by_id[doc_id]


def test_run_pyvi_skips_when_estimate_too_long(tmp_path, monkeypatch):
    pytest.importorskip("bm25s")
    pytest.importorskip("zstandard")
    dp = tmp_path / "docs.jsonl.zst"
    _write_docs(dp, [{"id": 1, "text": "sốt xuất huyết"}])
    qp = tmp_path / "q.jsonl"
    qp.write_text('{"id": 1, "query": "sốt"}\n', encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(bm25_chunk, "count_docs", lambda paths: 10_000_000)
    monkeypatch.setattr(bm25_chunk, "bench_tokenizer",
                        lambda paths, n, tokfn, limit=0: (2000, 6_000_000, 3600.0))
    assert run([str(dp)], str(qp), "pyvi", out, pyvi_max_hours=4.0) is None
    assert not (out / "cache_bm25c_pyvi.json").exists()


def test_report_overlap_and_hashes(tmp_path, capsys):
    cache = {"1": [[5, 0, 1.0], [6, 1, 0.9]], "2": [[1, 0, 1.0], [2, 1, 0.8]]}
    ref = {"1": [[6, 0, 1.0], [9, 1, 0.9]], "2": [[1, 0, 1.0], [3, 1, 0.8]]}
    cp = tmp_path / "c.json"
    rp = tmp_path / "r.json"
    cp.write_text(json.dumps(cache), encoding="utf-8")
    rp.write_text(json.dumps(ref), encoding="utf-8")
    f = tmp_path / "sub.zip"
    f.write_bytes(b"abc")
    report(str(cp), str(rp), 30, [str(f)])
    out = capsys.readouterr().out
    assert "overlap top-30" in out
    assert "1.00 doc/câu" in out
    assert "sha256 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" in out
    assert "3 bytes" in out
