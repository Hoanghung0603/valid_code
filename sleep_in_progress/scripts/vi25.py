"""vi25 = vi24 + goi VI 20261023 (43.154 doc); dieu phoi tai dung vi23/vi24 + cac cong cu co san.

Chay tung buoc qua deploy/b660_vi25.bat hoac deploy/z87_vi25.bat (scheduled task).
Khong nop bai, khong luu credential. Goi nguon va moi artefact vi24 deu CHI DOC.

Khac vi24:
  - nguon = handoff_vi_20261023 (3 shard, 43.154 rec); dedup voi chi muc vi24 + loai id
    ngoai tap VI cua kho links_corpus_vi_en_zh.parquet.
  - 2 zip cung mot vi25 (cung doclist, khac delta chunk): tran 93MB (so mot bien voi #6720)
    va tran 99MB; delta LON NHAT voi zip LZMA that <= tran (check-zip/next_delta + buoc +1).
  - them submission_vi25_newonly30.zip (30 doc moi/cau theo fhyde_v25, khong chunk).
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import vi23 as core

from vi24 import (fingerprint, read, write, run, vi, owned_lock, probe, guard_probe,
                  gpu_run, missing_docwin, merge_chunks)

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
S = ROOT / "scripts"
W = ROOT / "data/sub/diag25"
B = ROOT / "data/sub/diag24"
IX = ROOT / "data/index/vi25"
BASE_IX = ROOT / "data/index/vi24"
G = ROOT / "work/vi25"
Q8 = ROOT / "data/sub/qwen8b/cache_qwen8b_sorted.json"
REF_K = ROOT / "data/sub/diag14/submission_c120in.json"
REF_C = ROOT / "data/sub/diag11/chunks_rrf3d2_cf4.json"
P3 = ROOT / "data/sub/diag18/docwin_p3"
LINKS = ROOT / "data/sub/diag19/links_corpus_vi_en_zh.parquet"
SOURCE = ROOT / "data/crawl/handoff_vi_20261023"
SHARDS = ["vi-recov-00000.jsonl.zst", "vi-recov-00001.jsonl.zst", "vi-recov-00002.jsonl.zst"]
N_SOURCE = 43154
SANITY_DELTA = 46  # delta cua submission_vi24 (report_chain_vi24.json) - sanity phai trung tuyet doi
DELTA0 = 48        # diem bat dau nhu vi24
ZIPS = [("vi25_z93", 93_000_000), ("vi25_z99", 99_000_000)]


def corpus_vi_ids(path):
    """(vi_ids, all_ids) cua kho links_corpus_vi_en_zh; tap VI phan loai theo host URL (coverage21)."""
    import urllib.parse
    import pyarrow.parquet as pq
    from coverage21 import classify_host
    d = pq.read_table(path, columns=["id", "url"]).to_pydict()
    all_ids, vi_ids = set(), set()
    for i, u in zip(d["id"], d["url"]):
        i = int(i)
        all_ids.add(i)
        if classify_host(urllib.parse.urlparse(u).netloc) == "vi":
            vi_ids.add(i)
    return vi_ids, all_ids


def prep_records(records, old_ids, old_bp, vi_ids, all_ids=None):
    """Bo id trung chi muc vi24 + id ngoai tap VI kho; ghep boilerplate (host cu giu nguyen).

    Giu ban ghi nguyen van (raw text) ke ca khi BP lam sach rong - giong vi24.prep_records.
    """
    input_ids = {int(r["id"]) for r in records if r.get("id") is not None}
    overlap = sorted(input_ids & set(old_ids))
    outside = sorted(input_ids - set(vi_ids) - set(old_ids))
    drop = set(overlap) | set(outside)
    eligible = [r for r in records if r.get("id") is not None and int(r["id"]) not in drop]
    kept, stats, ids = core.filter_records(eligible, set())
    bp, bp_report = core.fit_new_hosts(old_bp, kept)
    from boilerplate import BoilerplateFilter
    filt = BoilerplateFilter.from_json(json.dumps(bp))
    empty = [int(r["id"]) for r in kept
             if not filt.clean(r.get("host", ""), r.get("text") or "").strip()]
    rep = {"records": len(records), "unique_input_ids": len(input_ids),
           "overlap_vi24": len(overlap), "overlap_ids": overlap,
           "ngoai_tap_vi": len(outside), "ngoai_tap_vi_ids": outside,
           "new": len(ids), "empty_after_boilerplate": len(empty),
           "empty_ids": empty, "filter": stats, "boilerplate": bp_report}
    if all_ids is not None:
        s = set(all_ids)
        rep["ngoai_kho"] = sum(1 for i in outside if i not in s)
        rep["trong_kho_khong_phai_vi"] = len(outside) - rep["ngoai_kho"]
    return kept, bp, rep


def stage_prep():
    import numpy as np
    import zstandard
    dest = W / "shards"
    dest.mkdir(parents=True, exist_ok=True)
    for name in SHARDS + ["manifest.sha256", "summary.json"]:
        target = dest / name
        if target.exists():
            assert fingerprint(target)["sha256"] == fingerprint(SOURCE / name)["sha256"], name
        else:
            shutil.copy2(SOURCE / name, target)
    vi("manifest", "--shards", dest, "--report", W / "manifest_report.json")
    rows = []
    for name in SHARDS:
        rows.extend(core.read_shard_records(dest / name))
    assert len(rows) == N_SOURCE, len(rows)
    vi_ids, all_ids = corpus_vi_ids(LINKS)
    kept, bp, report = prep_records(rows, set(map(int, np.load(BASE_IX / "ids.npy"))),
                                    read(BASE_IX / "boilerplate.json"), vi_ids, all_ids)
    filtered = W / "indexed"
    filtered.mkdir(exist_ok=True)
    with (filtered / "vi25-new.jsonl.zst").open("wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            for r in kept:
                z.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    write(W / "new_ids.json", sorted(int(r["id"]) for r in kept))
    write(W / "new_host.json", {str(r["id"]): r.get("host", "") for r in kept})
    write(IX / "boilerplate.json", bp)
    write(W / "prep_report.json", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("overlap_ids", "ngoai_tap_vi_ids", "empty_ids")},
                     ensure_ascii=False), flush=True)


def chain_args(base, cands, name, mode="that", delta=DELTA0):
    suffix = "v24" if base == B else "v25"
    return ["chain", "--mode", mode, "--name", name,
            "--pool", base / f"cache_pool_{suffix}.json", "--q8b", Q8,
            "--dense", base / f"dense_{suffix}_top200.json",
            "--hyde", base / f"hyde_{suffix}_top200.json", "--cands", cands,
            "--ref-kq", REF_K, "--ref-chunks", REF_C, "--delta", delta, "--out-dir", W]


def stage_sanity():
    vi(*chain_args(B, B / "chunk_dw_u900_v24.json.zst", "km", "khong-moi", delta=SANITY_DELTA),
       "--ref-sub", B / "submission_vi24.json")
    for name in ["rrf3d2", "fhyde"]:
        vi("cmp-exact", "--a", W / f"cache_{name}_khongmoi.json", "--b", B / f"cache_{name}_v24.json",
           "--report", W / f"cmp_{name}.json")


def stage_index():
    from bigindex import build
    old_meta = read(BASE_IX / "meta.json")
    paths = [Path(p) for p in old_meta["shards"]] + [W / "indexed/vi25-new.jsonl.zst"]
    # bigindex sorts paths: verify its resulting prefix before doing expensive work.
    assert sorted(paths)[:len(old_meta["shards"])] == [Path(p) for p in old_meta["shards"]]
    build(paths, IX, lang=old_meta["lang"], nbits=old_meta["nbits"], workers=3,
          budget=20_000_000, log=lambda m: print(m, flush=True))
    vi("verify-idx", "--index-old-ids", BASE_IX / "ids.npy", "--index-new-ids", IX / "ids.npy",
       "--new-ids", W / "new_ids.json", "--out", W / "new_idx.json", "--report", W / "verify_idx_report.json")
    stage_exports()


def stage_exports():
    for mode in ["embed", "docwin"]:
        vi("export-new", "--shards", W / "indexed", "--bp", IX / "boilerplate.json",
           "--idx", W / "new_idx.json", "--mode", mode, "--out", W / f"docs_vi25_{mode}.jsonl.zst",
           "--report", W / f"export_{mode}_report.json")


def stage_bm25():
    run("search_cache.py", "--index", IX, "--queries", ROOT / "data/query.parquet",
        "--out", W / "cache_vi25_k300.json", "-k", 300)
    vi("bm25-diff", "--cache-old", B / "cache_vi24_k300.json", "--cache-new", W / "cache_vi25_k300.json",
       "--new-ids", W / "new_ids.json", "--report", W / "bm25_diff.json")


def stage_embed():
    docs = G / "docs_embed"
    docs.mkdir(exist_ok=True)
    src = G / "docs_vi25_embed.jsonl.zst"
    if src.exists() and not (docs / src.name).exists():
        shutil.move(str(src), str(docs / src.name))
    (G / "empty_emb").mkdir(exist_ok=True)
    gpu_run("embed_all.py", "--docs", docs, "--work-old", G / "empty_emb", "--work", G / "emb",
            "--queries", ROOT / "work/queries.jsonl", "--out", G / "dense_new_top200.json",
            "--doc-len", 512, "--query-len", 256, "--batch", 16, "--top", 200)
    gpu_run("dense_topk.py", "--emb", G / "emb", "--docs", docs / "docs_vi25_embed.jsonl.zst",
            "--queries", ROOT / "work/hyde/q_hyde.jsonl", "--query-len", 512, "--top", 200,
            "--device", "cuda", "--out", G / "hyde_new_top200.json")


def stage_pool():
    for kind in ["dense", "hyde"]:
        vi("dense-merge", "--cache-old", B / f"{kind}_v24_top200.json",
           "--cache-new", W / f"{kind}_new_top200.json", "--top", 200,
           "--out", W / f"{kind}_v25_top200.json", "--report", W / f"{kind}_merge_report.json")
    vi("pool-new", "--pool", B / "cache_pool_v24.json", "--bm25-new", W / "cache_vi25_k300.json",
       "--dense-new", W / "dense_v25_top200.json", "--new-ids", W / "new_ids.json",
       "--idx", W / "new_idx.json", "--out-cands", W / "cands_rr_new.json",
       "--out-new-hits", W / "new_hits.json", "--report", W / "pool_new_report.json")


def stage_rr():
    gpu_run("rerank_run.py", "--docs", G / "docs_embed/docs_vi25_embed.jsonl.zst",
            "--queries", ROOT / "work/queries.jsonl", "--cands", G / "cands_rr_new.json",
            "--out", G / "rerank_new.json", "--max-len", 512, "--batch", 16, "--top", 500,
            "--resume-dir", G / "rr_parts")


def stage_doclist():
    import numpy as np
    vi("pool-merge", "--pool-old", B / "cache_pool_v24.json", "--new-hits", W / "new_hits.json",
       "--rerank", W / "rerank_new.json", "--out", W / "cache_pool_v25.json", "--report", W / "pool_merge_report.json")
    assert read(W / "pool_merge_report.json")["hit_logit_None"] == 0
    vi("fuse", "--pool", W / "cache_pool_v25.json", "--q8b", Q8, "--dense", W / "dense_v25_top200.json",
       "--hyde", W / "hyde_v25_top200.json", "--out-rrf3d2", W / "cache_rrf3d2_v25.json",
       "--out-fhyde", W / "cache_fhyde_v25.json", "--report", W / "fuse_report.json")
    vi("doclist", "--pool", W / "cache_pool_v25.json", "--fhyde", W / "cache_fhyde_v25.json",
       "--ref-chunks", REF_C, "--out", W / "doclist_v25.json", "--report", W / "doclist_report.json")
    dl = read(W / "doclist_v25.json")
    idx = {str(d): i for i, d in enumerate(np.load(IX / "ids.npy"))}
    cands = missing_docwin(dl, read(B / "chunk_dw_u900_v24.json.zst"), read(B / "doclist_v24.json"), idx)
    new_ids = set(map(str, read(W / "new_ids.json")))
    # Previously scored p3 pairs can be reused even when absent from vi24's doclist.
    p3 = read(P3 / "docwin_p3.json.zst")
    reuse, fresh = {}, {}
    for q, rows in cands.items():
        for h in rows:
            target = reuse if str(h[0]) in p3.get(q, {}) else fresh
            target.setdefault(q, []).append(h)
    write(W / "cands_docwin_vi25.json", fresh)
    write(W / "cands_docwin_reuse.json", reuse)
    write(W / "cache_docwin_old.json", {q: [[*h, 0.0] for h in rows if str(h[0]) not in new_ids]
                                           for q, rows in fresh.items()})
    run("export_docs.py", "--index", BASE_IX, "--cache", W / "cache_docwin_old.json",
        "--n", 1000, "--chars", 200000, "--no-title", "--part-mb", 80,
        "--out-docs", W / "docs_vi25_old.jsonl.zst", "--out-cands", W / "export_old_cands.json")
    n_new = sum(str(h[0]) in new_ids for rows in fresh.values() for h in rows)
    write(W / "docwin_cands_report.json", {"pairs": sum(map(len, fresh.values())), "new_doc_pairs": n_new,
          "old_doc_pairs": sum(map(len, fresh.values())) - n_new, "p3_reused_pairs": sum(map(len, reuse.values()))})


def stage_docwin():
    gpu_run("dwx_p3.py", "winsample", "--docwin", ROOT / "work/docwin_ext/docwin_ext.json.zst",
            "--docs", ROOT / "work/docwin_ext/docwin_ext.part000.jsonl.zst",
            ROOT / "work/docwin_ext/docwin_ext.part001.jsonl.zst", "--queries", ROOT / "work/queries.jsonl",
            "--out", G / "winsample.json", "--n", 300)
    gpu_run("dwx_p3.py", "score", "--in", G / "winsample.json", "--out", G / "winscore.json",
            "--model", "BAAI/bge-reranker-v2-m3", "--max-len", 512, "--batch", 16)
    gpu_run("dwx_p3.py", "wincheck", "--ref", G / "winsample.json", "--got", G / "winscore.json",
            "--report", G / "wincheck.json")
    docs = [G / "docs_vi25_docwin.jsonl.zst"] + sorted(G.glob("docs_vi25_old*.jsonl.zst"))
    gpu_run("docwin_run.py", "--docs", *docs, "--queries", ROOT / "work/queries.jsonl",
            "--cands", G / "cands_docwin_vi25.json", "--out", G / "docwin_vi25.json.zst",
            "--max-len", 512, "--batch", 16, "--top-docs", 500, "--window-words", 300,
            "--window-stride", 250, "--resume-dir", G / "dw_parts")


def stage_chunk():
    from chunks_from_docwin import build_q_rows, select_q_rows, verify_verbatim
    from dwx_merge import write_zst_stream
    from rerank_run import load_docs
    cands, reuse = read(W / "cands_docwin_vi25.json"), read(W / "cands_docwin_reuse.json")
    wins = read(W / "docwin_vi25.json.zst")
    p3 = read(P3 / "docwin_p3.json.zst")
    for q, rows in reuse.items():
        cands.setdefault(q, []).extend(rows)
        wins.setdefault(q, {}).update({str(h[0]): p3[q][str(h[0])] for h in rows})
    paths = [W / "docs_vi25_docwin.jsonl.zst", *sorted(W.glob("docs_vi25_old*.jsonl.zst")),
             P3 / "docwin_p3.part000.jsonl.zst", P3 / "docwin_p3.part001.jsonl.zst"]
    wanted = {h[0] for rows in cands.values() for h in rows}
    docs = load_docs(list(map(str, paths)), wanted=wanted)
    missing = wanted - set(docs)
    assert not missing, f"missing docwin texts: {len(missing)}"
    absent = [(q, h[0]) for q, rows in cands.items() for h in rows
              if str(h[0]) not in wins.get(q, {}) and docs[h[0]].strip()]
    assert not absent, f"missing nonempty docwin scores: {absent[:5]}"
    cache = {}
    new = {q: select_q_rows(build_q_rows(rows, docs, wins.get(q, {}), 900, 500, cache), 100000)
           for q, rows in cands.items()}
    bad = verify_verbatim(new, docs)
    assert bad == 0
    old = read(B / "chunk_dw_u900_v24.json.zst")
    merged = merge_chunks(old, new)
    write_zst_stream(W / "chunk_dw_u900_v25.json.zst", merged)
    write(W / "chunk_report.json", {"new_chunks": sum(map(len, new.values())), "verbatim_bad": bad,
          "empty_text_pairs": sum(not docs[h[0]].strip() for rows in cands.values() for h in rows),
          "old_chunks": sum(map(len, old.values())), "total_chunks": sum(map(len, merged.values()))})


def chain_zip_check(name, delta, max_zip, validate=False):
    """Chay chain voi --delta, (tuy chon) validate, zip LZMA, check_zip. Tra ket qua mot lan thu."""
    vi(*chain_args(W, W / "chunk_dw_u900_v25.json.zst", name, delta=delta))
    if validate:
        run("evaluate.py", "validate", W / f"submission_{name}.json")
    run("zip_submission.py", "--json", W / f"submission_{name}.json",
        "--zip", W / f"submission_{name}.zip", "--method", "lzma")
    code = core.check_zip(str(W / f"submission_{name}.json"), str(W / f"submission_{name}.zip"),
                          str(W / f"report_chain_{name}.json"), name, max_zip=max_zip)
    rep = read(W / f"report_chain_{name}.json")
    entry = rep.get("zip", {}).get(name, {})
    return {"code": code, "next_delta": entry.get("next_delta"), "zip_bytes": entry.get("zip_bytes"),
            "json_bytes": rep.get("json_bytes"), "tong_chunk": rep.get("tong_chunk"),
            "hist_kcap": rep.get("hist_kcap")}


def find_delta(max_zip, delta0, probe):
    """Delta LON NHAT voi zip LZMA THAT <= max_zip (chay bang ten phu, khong anh huong ket qua that).

    Giam bang vong check-zip/next_delta san co (giong vi24) den khi duoi tran, roi buoc uoc
    luong cx110.largest_delta_within + nhi phan/+1 xac nhan bang zip that. Tong chunk khong
    tang nua = bao hoa |C_q| -> dung (delta lon hon cho cung noi dung).
    """
    from cx130 import largest_delta_within
    tried = {}

    def attempt(delta):
        if delta not in tried:
            tried[delta] = chain_zip_check(probe, delta, max_zip)
        return tried[delta]

    def fits(delta):
        return attempt(delta)["code"] == 0

    def grew(delta, base):
        return tried[delta].get("tong_chunk") != tried[base].get("tong_chunk")

    lo = delta0
    while not fits(lo):
        r = tried[lo]
        assert r["code"] == 3 and r["next_delta"] is not None and 0 <= r["next_delta"] < lo, r
        lo = r["next_delta"]
    hist = next(iter(tried.values()))["hist_kcap"]
    for _ in range(3):
        r = tried[lo]
        up = int(largest_delta_within(hist, r["json_bytes"], r["tong_chunk"], lo + 500, max_zip)) \
            if hist and r.get("tong_chunk") and r.get("json_bytes") else lo + 1
        up = max(up, lo + 1)
        if fits(up) and grew(up, lo):
            lo = up
            continue
        if fits(up):
            break  # bao hoa |C_q|: delta lon hon van cho cung noi dung, dung o lo
        hi = up  # vuot tran: tim delta lon nhat trong [lo, hi] bang zip that
        while hi - lo > 1 and len(tried) < 16:
            mid = (lo + hi) // 2
            if fits(mid) and grew(mid, lo):
                lo = mid
            else:
                hi = mid
        break
    while lo + 1 not in tried and len(tried) < 16:  # buoc +1 cuoi: dam bao lo+1 khong duoi tran
        if fits(lo + 1) and grew(lo + 1, lo):
            lo += 1
        else:
            break
    return lo, [{"delta": k, "code": v["code"], "zip_bytes": v["zip_bytes"]} for k, v in sorted(tried.items())]


def stage_build():
    attempts = {}
    deltas = {}
    for name, max_zip in ZIPS:
        deltas[name], attempts[name] = find_delta(max_zip, DELTA0, "vi25p" + name[-2:])
    for name, max_zip in ZIPS:
        code = chain_zip_check(name, deltas[name], max_zip, validate=True)["code"]
        assert code == 0, (name, code)
    dl93 = read(W / "doclist_vi25_z93.json")
    dl99 = read(W / "doclist_vi25_z99.json")
    same = dl93 == dl99
    assert same, "doclist z93/z99 phai giong nhau (chi khac delta chunk)"
    write(W / "build_report.json", {"deltas": deltas, "attempts": attempts,
          "zips": dict(ZIPS), "same_doclist": same})
    stage_stats(deltas)


def stage_stats(deltas=None):
    if deltas is None:
        deltas = {name: (read(W / f"report_chain_{name}.json") or {}).get("delta") for name, _ in ZIPS}
    table = []
    for name, max_zip in ZIPS:
        vi("stats", "--submission", W / f"submission_{name}.json", "--fhyde", W / "cache_fhyde_v25.json",
           "--fhyde-old", B / "cache_fhyde_v24.json", "--pool", W / "cache_pool_v25.json",
           "--new-ids", W / "new_ids.json", "--links", W / "new_host.json",
           "--report", W / f"stats_report_{name}.json")
        st = read(W / f"stats_report_{name}.json")
        fp = fingerprint(W / f"submission_{name}.zip")
        table.append({"zip": f"submission_{name}.zip", "bytes": fp["bytes"], "sha256": fp["sha256"],
                      "delta": deltas.get(name), "max_zip": max_zip, "validate": "ok",
                      "TB_doc_cau": st["TB_doc_cau"], "TB_doc_moi_cau": st["TB_doc_moi_cau"],
                      "TB_chunk_cau": st["TB_chunk_cau"],
                      "TB_chunk_tu_doc_moi_cau": st["TB_chunk_tu_doc_moi_cau"]})
    vi("newonly30", "--fhyde", W / "cache_fhyde_v25.json", "--new-ids", W / "new_ids.json",
       "--out", W / "submission_vi25_newonly30.json", "--report", W / "newonly30_report.json")
    run("evaluate.py", "validate", W / "submission_vi25_newonly30.json")
    run("zip_submission.py", "--json", W / "submission_vi25_newonly30.json",
        "--zip", W / "submission_vi25_newonly30.zip", "--method", "lzma")
    vi("stats", "--submission", W / "submission_vi25_newonly30.json", "--fhyde", W / "cache_fhyde_v25.json",
       "--fhyde-old", B / "cache_fhyde_v24.json", "--pool", W / "cache_pool_v25.json",
       "--new-ids", W / "new_ids.json", "--links", W / "new_host.json",
       "--report", W / "stats_report_newonly30.json")
    st = read(W / "stats_report_newonly30.json")
    fp = fingerprint(W / "submission_vi25_newonly30.zip")
    table.append({"zip": "submission_vi25_newonly30.zip", "bytes": fp["bytes"], "sha256": fp["sha256"],
                  "delta": None, "max_zip": None, "validate": "ok",
                  "TB_doc_cau": st["TB_doc_cau"], "TB_doc_moi_cau": st["TB_doc_moi_cau"],
                  "TB_chunk_cau": st["TB_chunk_cau"],
                  "TB_chunk_tu_doc_moi_cau": st["TB_chunk_tu_doc_moi_cau"]})
    write(W / "zip_table.json", table)
    write(W / "zip_fingerprints.json", [fingerprint(W / row["zip"]) for row in table])


def vinmec_stats(work):
    times = []
    for p in sorted(work.glob("*.log")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("vinmec "):
                try:
                    times.append(float(line.split()[2]))
                except (IndexError, ValueError):
                    pass
    return {"so_lan": len(times), "max_s": max(times) if times else None}


def status(machine):
    work = G if machine == "z87" else W
    print("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    for p in sorted(work.glob("*.log")):
        with p.open("rb") as f:
            f.seek(max(0, p.stat().st_size - 5000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        print(p.name, "mtime", time.ctime(p.stat().st_mtime), "\n", "\n".join(lines[-8:]))
    print("locks", {p.name: p.read_text(errors="replace") for p in (ROOT / "locks").glob("*.lock")})
    print("vinmec", vinmec_stats(work))
    tasks = subprocess.run(["schtasks", "/Query", "/FO", "CSV", "/NH"], capture_output=True, text=True)
    print("tasks", "\n".join(l for l in tasks.stdout.splitlines() if "vi25" in l))
    print("files", json.dumps({p.name: p.stat().st_size for p in work.glob("*") if p.is_file()}))


def reports(machine):
    work = G if machine == "z87" else W
    for name in ["prep_report.json", "verify_idx_report.json", "report_chain_km.json",
                 "pool_new_report.json", "pool_merge_report.json", "doclist_report.json",
                 "docwin_cands_report.json", "chunk_report.json", "report_chain_vi25_z93.json",
                 "report_chain_vi25_z99.json", "build_report.json", "stats_report_vi25_z93.json",
                 "stats_report_vi25_z99.json", "stats_report_newonly30.json",
                 "newonly30_report.json", "zip_table.json", "zip_fingerprints.json", "wincheck.json"]:
        p = work / name
        if p.exists():
            obj = read(p)
            if isinstance(obj, dict):
                obj = {k: v for k, v in obj.items()
                       if k not in ["hist_kcap", "overlap_ids", "empty_ids", "ngoai_tap_vi_ids", "attempts"]}
            print(name, json.dumps(obj, ensure_ascii=False), flush=True)
    print("vinmec", json.dumps(vinmec_stats(work)), flush=True)
    if machine == "b660" and (IX / "meta.json").exists():
        meta = read(IX / "meta.json")
        print("index", json.dumps({k: v for k, v in meta.items() if k != "shards"}))
        print("index_bytes", sum(p.stat().st_size for p in IX.iterdir() if p.is_file()))
        if (W / "cache_pool_v25.json").exists():
            from union_sub import adaptive_n
            for label, path in [("vi24", B / "cache_pool_v24.json"), ("vi25", W / "cache_pool_v25.json")]:
                pool = read(path)
                print("pool", label, "mean", sum(map(len, pool.values())) / len(pool),
                      "n_q_mean", sum(adaptive_n(h, -2.0, 40, 400) for h in pool.values()) / len(pool))


def cleanup(machine):
    lock = ROOT / "locks" / ("gpu.lock" if machine == "z87" else "heavy_job.lock")
    if lock.exists() and "vi25" in lock.read_text(errors="replace"):
        raise RuntimeError("vi25 lock still exists; inspect job before cleanup")
    stages = ["embed", "rr", "docwin"] if machine == "z87" else \
             ["prep", "index", "sanity", "bm25", "pool", "doclist", "chunk", "build"]
    for stage in stages:
        name = "r2ai_vi25_" + stage
        query = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "CSV", "/NH"],
                               capture_output=True, text=True)
        if query.returncode:
            continue
        assert '"Running"' not in query.stdout, name + " still running"
        subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], check=True)
    print("CLEANUP DONE", machine, "locks", {p.name: p.read_text(errors="replace")
                                             for p in (ROOT / "locks").glob("*.lock")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prep", "sanity", "index", "exports", "bm25", "embed",
                                          "pool", "rr", "doclist", "docwin", "chunk", "build",
                                          "stats", "status", "reports", "cleanup"])
    parser.add_argument("--machine", choices=["b660", "z87"], default="b660")
    args = parser.parse_args()
    if args.stage == "status":
        status(args.machine)
        return
    if args.stage == "reports":
        reports(args.machine)
        return
    if args.stage == "cleanup":
        cleanup(args.machine)
        return
    work = G if args.machine == "z87" else W
    work.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "locks" / ("gpu.lock" if args.machine == "z87" else "heavy_job.lock")
    with owned_lock(lock, f"vi25 {args.stage} pid={os.getpid()}"):
        globals()["stage_" + args.stage]()
    print("DONE", args.stage, flush=True)


if __name__ == "__main__":
    main()
