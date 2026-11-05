import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from candgen14 import (  # noqa: E402
    build_ceil2,
    build_newonly30,
    hit_ids,
    new_candidates,
    order_new,
    per_query_new,
    rrf_score,
    source_ranks,
    stats_new,
    write_new_cands,
    zipcheck,
)


def test_hit_ids_and_source_ranks_dedup():
    hits = [[5, 0, 1.0], [3, 1, 0.9], [5, 2, 0.8], [7, 3, 0.7]]
    assert hit_ids(hits) == [5, 3, 7]
    assert hit_ids(hits, limit=2) == [5, 3]
    assert source_ranks(hits) == {5: 1, 3: 2, 7: 4}
    assert source_ranks(hits, limit=3) == {5: 1, 3: 2}


def test_new_candidates_excludes_pool_and_keeps_ranks():
    pool = [[1, 0, 9.0], [2, 0, 8.0]]
    sources = {
        "pool2": [[1, 0, 9.0], [9, 0, 8.0], [2, 0, 7.0], [8, 0, 6.0]],
        "hyde": [[8, 0, 0.9], [7, 0, 0.8]],
    }
    out = new_candidates(pool, sources)
    assert set(out) == {9, 8, 7}
    assert out[9] == {"pool2": 2}
    assert out[8] == {"pool2": 4, "hyde": 1}
    assert out[7] == {"hyde": 2}


def test_rrf_score_and_order_new():
    assert abs(rrf_score({"a": 1}, k=60) - 1 / 61) < 1e-12
    assert abs(rrf_score({"a": 1, "b": 2}, k=60) - (1 / 61 + 1 / 62)) < 1e-12
    new_map = {
        10: {"pool2": 5},                                   # 1/65
        20: {"pool2": 10, "hyde": 20},                      # 1/70 + 1/80
        30: {"hyde": 5},                                    # 1/65 = doc 10, tie
    }
    rows = order_new(new_map, k=60)
    order = [d for d, _, _ in rows]
    assert order[0] == 20
    assert order[1] == 10 and order[2] == 30
    assert rows[0][1] == round(1 / 70 + 1 / 80, 8)
    assert rows[1][2] == [["pool2", 5]]
    assert rows[2][2] == [["hyde", 5]]


def test_order_new_multi_source_first_and_src_order():
    new_map = {1: {"bm25": 1, "pool2": 1}, 2: {"dense": 1, "hyde": 1}}
    rows = order_new(new_map, k=60)
    assert [r[0] for r in rows] == [1, 2]
    assert rows[0][2] == [["pool2", 1], ["bm25", 1]]
    assert rows[1][2] == [["hyde", 1], ["dense", 1]]


def test_stats_new_counts():
    per_q = {
        "1": {10: {"pool2": 2}, 20: {"pool2": 3, "hyde": 4}, 30: {"hyde": 60}},
        "2": {40: {"pool2": 10}, 50: {"dense": 250}},
    }
    rep = stats_new(per_q)
    assert rep["so_cau"] == 2
    assert rep["TB_new_cau"] == 2.5
    assert rep["min_new"] == 2 and rep["max_new"] == 3 and rep["tong_new"] == 5
    assert rep["nguon"]["pool2"]["TB_new_co_mat"] == 1.5
    assert rep["nguon"]["pool2"]["TB_new_rieng"] == 1.0
    assert rep["nguon"]["hyde"]["TB_new_rieng"] == 0.5
    assert rep["nguon"]["hyde"]["TB_new_co_mat"] == 1.0
    assert rep["nguon"]["dense"]["TB_new_rieng"] == 0.5
    assert rep["nguon"]["bm25"]["TB_new_rieng"] == 0.0
    assert rep["trung_cap_nguon"]["pool2|hyde"] == 0.5
    assert rep["nguon"]["hyde"]["phan_bo_hang"]["1-50"] == 0.5
    assert rep["nguon"]["hyde"]["phan_bo_hang"]["51-100"] == 0.5
    assert rep["nguon"]["pool2"]["min_hang"] == 2
    assert rep["nguon"]["pool2"]["max_hang"] == 10


def test_per_query_new_and_write_new_cands(tmp_path):
    pool = {"1": [[1, 0, 1.0]], "2": [[2, 0, 1.0]]}
    sources = [{"1": [[1, 0, 1.0], [9, 0, 0.5]], "2": []},
               {"1": [], "2": [[8, 0, 0.5], [2, 0, 0.4]]}]
    per_q = per_query_new(pool, sources)
    assert set(per_q["1"]) == {9} and set(per_q["2"]) == {8}
    rows_q = {q: order_new(m) for q, m in per_q.items()}
    p1, p2 = write_new_cands(per_q, rows_q, tmp_path)
    cands = json.loads(p1.read_text(encoding="utf-8"))
    assert cands == {"1": [9], "2": [8]}
    meta = json.loads(p2.read_text(encoding="utf-8"))
    assert meta["1"][0]["doc_id"] == 9 and meta["1"][0]["n_src"] == 1


def test_build_ceil2_pool_before_new_chunks_copied():
    cx = [{"id": 1, "relevant_docs": [1, 2],
           "relevant_chunks": [{"doc_id": 2, "chunk_text": "abc"}]}]
    pool = {"1": [[1, 0, 9.0], [2, 0, 8.0], [3, 0, 7.0]]}
    new_rows = {"1": [(9, 0.0, []), (8, 0.0, [])]}
    out, st = build_ceil2(cx, pool, new_rows)
    assert out[0]["relevant_docs"] == [1, 2, 3, 9, 8]
    assert out[0]["relevant_chunks"] == [{"doc_id": 2, "chunk_text": "abc"}]
    assert st["TB_doc_them_cho_chunk"] == 0.0


def test_build_ceil2_appends_missing_chunk_doc():
    cx = [{"id": 7, "relevant_docs": [],
           "relevant_chunks": [{"doc_id": 99, "chunk_text": "x"}]}]
    pool = {"7": [[1, 0, 1.0]]}
    out, st = build_ceil2(cx, pool, {"7": [(5, 0.0, [])]})
    assert out[0]["relevant_docs"] == [1, 5, 99]
    assert st["tong_doc_them_cho_chunk"] == 1


def test_build_newonly30_limits_and_empty_chunks():
    new_rows = {str(q): [(q * 100 + i, 0.0, []) for i in range(35)] for q in (1, 2)}
    out = build_newonly30(new_rows, n_docs=30)
    assert [s["id"] for s in out] == [1, 2]
    assert len(out[0]["relevant_docs"]) == 30
    assert out[0]["relevant_docs"] == [100 + i for i in range(30)]
    assert out[0]["relevant_chunks"] == []
    short = build_newonly30({"5": [(1, 0.0, [])]}, n_docs=30)
    assert short[0]["relevant_docs"] == [1]


def test_zipcheck_ok_and_bad(tmp_path):
    good = tmp_path / "good.zip"
    src = tmp_path / "submission.json"
    src.write_text("[]", encoding="utf-8")
    with zipfile.ZipFile(good, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(src, arcname="submission.json")
    rep = zipcheck(good)
    assert rep["ok"] and rep["so_entry"] == 1 and rep["ten_dung"] and rep["doc_lai"]
    assert len(rep["sha256"]) == 64

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(src, arcname="submission.json")
        z.write(src, arcname="extra.json")
    rep2 = zipcheck(bad, max_zip=10 ** 9)
    assert not rep2["ok"] and rep2["so_entry"] == 2
    rep3 = zipcheck(good, max_zip=1)
    assert not rep3["ok"] and not rep3["duoi_gioi_han"]


def test_cli_stats_and_subs_and_zipcheck(tmp_path, capsys):
    import candgen14

    def write(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj), encoding="utf-8")
        return p

    pool = write("pool.json", {"1": [[1, 0, 9.0], [2, 0, 8.0]]})
    p2 = write("pool2.json", {"1": [[1, 0, 9.0], [9, 0, 8.0]]})
    hyde = write("hyde.json", {"1": [[8, 0, 0.9], [1, 0, 0.8]]})
    reason = write("reason.json", {"1": [[8, 0, 0.9]]})
    dense = write("dense.json", {"1": [[7, 0, 0.7]]})
    bm25 = write("bm25.json", {"1": [[2, 0, 5.0], [6, 0, 4.0]]})
    out_dir = tmp_path / "diag18"
    out_dir.mkdir()
    args = ["stats", "--pool", str(pool), "--pool2", str(p2), "--hyde", str(hyde),
            "--reason", str(reason), "--dense", str(dense), "--bm25", str(bm25),
            "--out-dir", str(out_dir)]
    with pytest.raises(SystemExit) as e:
        sys.argv = ["candgen14.py"] + args
        candgen14.main()
    assert e.value.code == 0
    cands = json.loads((out_dir / "new_cands.json").read_text(encoding="utf-8"))
    assert set(cands["1"]) == {9, 8, 7, 6}
    assert cands["1"][0] == 8

    cx = write("cx.json", [{"id": 1, "relevant_docs": [1, 2],
                            "relevant_chunks": [{"doc_id": 2, "chunk_text": "abc"}]}])
    args2 = ["subs", "--pool", str(pool), "--new", str(out_dir / "new_cands.json"),
             "--cx130", str(cx), "--out-dir", str(out_dir), "--n", "2"]
    with pytest.raises(SystemExit) as e2:
        sys.argv = ["candgen14.py"] + args2
        candgen14.main()
    assert e2.value.code == 0
    ceil2 = json.loads((out_dir / "submission_ceil2.json").read_text(encoding="utf-8"))
    assert ceil2[0]["relevant_docs"] == [1, 2, 8, 7, 6, 9]
    assert ceil2[0]["relevant_chunks"] == [{"doc_id": 2, "chunk_text": "abc"}]
    only = json.loads((out_dir / "submission_newonly30.json").read_text(encoding="utf-8"))
    assert only[0]["relevant_docs"] == [8, 7] and only[0]["relevant_chunks"] == []

    zp = out_dir / "submission_newonly30.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_LZMA) as z:
        z.write(out_dir / "submission_newonly30.json", arcname="submission.json")
    with pytest.raises(SystemExit) as e3:
        sys.argv = ["candgen14.py", "zipcheck", "--zip", str(zp)]
        candgen14.main()
    assert e3.value.code == 0
    assert "sha256" in capsys.readouterr().out
