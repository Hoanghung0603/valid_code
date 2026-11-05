"""vi24 = vi23 + handoff VI 20261021; orchestration reusing vi23 and existing tools.

Run stages through deploy/b660_vi24.bat or deploy/z87_vi24.bat (scheduled tasks).
No submission/network credentials. Source handoff and all vi23 artifacts are read-only.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

import vi23 as core

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
S = ROOT / "scripts"
W = ROOT / "data/sub/diag24"
B = ROOT / "data/sub/diag23"
IX = ROOT / "data/index/vi24"
BASE_IX = ROOT / "data/index/vi23"
G = ROOT / "work/vi24"
SOURCE_SHA = "712f7e076dc754ff49997ce3608c5222f79143c88249fb1eccf10234db81cc16"
Q8 = ROOT / "data/sub/qwen8b/cache_qwen8b_sorted.json"
REF_K = ROOT / "data/sub/diag14/submission_c120in.json"
REF_C = ROOT / "data/sub/diag11/chunks_rrf3d2_cf4.json"


def read(p):
    return core.load_json(p)


def write(p, obj):
    core.write_json(p, obj)


def fingerprint(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": str(p), "bytes": Path(p).stat().st_size, "sha256": h.hexdigest()}


def run(script, *args):
    cmd = [sys.executable, str(S / script), *map(str, args)]
    print("RUN", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True)


def vi(*args):
    run("vi23.py", *args)


@contextmanager
def owned_lock(path, label):
    """Atomic acquisition; never remove a lock belonging to another job."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(label)
    try:
        yield
    finally:
        if path.exists() and path.read_text(encoding="utf-8") == label:
            path.unlink()


def prep_records(records, old_ids, old_bp):
    """Dedup against actual vi23 IDs, retaining raw text even when BP cleans it empty."""
    input_ids = {int(r["id"]) for r in records if r.get("id") is not None}
    overlap = sorted(input_ids & set(old_ids))
    kept, stats, ids = core.filter_records(records, set(old_ids))
    bp, bp_report = core.fit_new_hosts(old_bp, kept)
    from boilerplate import BoilerplateFilter
    filt = BoilerplateFilter.from_json(json.dumps(bp))
    empty = [int(r["id"]) for r in kept if not filt.clean(r.get("host", ""), r.get("text") or "").strip()]
    return kept, bp, {"records": len(records), "unique_input_ids": len(input_ids),
                      "overlap_vi23": len(overlap), "overlap_ids": overlap,
                      "new": len(ids), "empty_after_boilerplate": len(empty),
                      "empty_ids": empty, "filter": stats, "boilerplate": bp_report}


def stage_prep():
    import numpy as np
    import zstandard
    source = ROOT / "data/crawl/handoff_vi_20261021"
    dest = W / "shards"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["vi-recov-00000.jsonl.zst", "manifest.sha256", "summary.json"]:
        target = dest / name
        if target.exists():
            assert fingerprint(target)["sha256"] == fingerprint(source / name)["sha256"]
        else:
            shutil.copy2(source / name, target)
    assert fingerprint(dest / "vi-recov-00000.jsonl.zst")["sha256"] == SOURCE_SHA
    vi("manifest", "--shards", dest, "--report", W / "manifest_report.json")
    rows = core.read_shard_records(dest / "vi-recov-00000.jsonl.zst")
    assert len(rows) == 16110
    kept, bp, report = prep_records(rows, set(map(int, np.load(BASE_IX / "ids.npy"))),
                                     read(BASE_IX / "boilerplate.json"))
    filtered = W / "indexed"
    filtered.mkdir(exist_ok=True)
    with (filtered / "vi24-new.jsonl.zst").open("wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            for r in kept:
                z.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    write(W / "new_ids.json", sorted(int(r["id"]) for r in kept))
    write(W / "new_host.json", {str(r["id"]): r.get("host", "") for r in kept})
    write(IX / "boilerplate.json", bp)
    write(W / "prep_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def chain_args(base, cands, name, mode="that", delta=48):
    suffix = "v23" if base == B else "v24"
    return ["chain", "--mode", mode, "--name", name,
            "--pool", base / f"cache_pool_{suffix}.json", "--q8b", Q8,
            "--dense", base / f"dense_{suffix}_top200.json",
            "--hyde", base / f"hyde_{suffix}_top200.json", "--cands", cands,
            "--ref-kq", REF_K, "--ref-chunks", REF_C, "--delta", delta, "--out-dir", W]


def stage_sanity():
    vi(*chain_args(B, B / "chunk_dw_u900_v23.json.zst", "km", "khong-moi"),
       "--ref-sub", B / "submission_vi23.json")
    for name in ["rrf3d2", "fhyde"]:
        vi("cmp-exact", "--a", W / f"cache_{name}_khongmoi.json", "--b", B / f"cache_{name}_v23.json",
           "--report", W / f"cmp_{name}.json")


def stage_index():
    from bigindex import build
    old_meta = read(BASE_IX / "meta.json")
    paths = [Path(p) for p in old_meta["shards"]] + [W / "indexed/vi24-new.jsonl.zst"]
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
           "--idx", W / "new_idx.json", "--mode", mode, "--out", W / f"docs_vi24_{mode}.jsonl.zst",
           "--report", W / f"export_{mode}_report.json")


def stage_bm25():
    run("search_cache.py", "--index", IX, "--queries", ROOT / "data/query.parquet",
        "--out", W / "cache_vi24_k300.json", "-k", 300)
    vi("bm25-diff", "--cache-old", B / "cache_vi23_k300.json", "--cache-new", W / "cache_vi24_k300.json",
       "--new-ids", W / "new_ids.json", "--report", W / "bm25_diff.json")


def probe():
    result = subprocess.run(["curl.exe", "-s", "--max-time", "5", "-o", "NUL", "-w",
                             "%{http_code} %{time_total}", "https://www.vinmec.com/"],
                            capture_output=True, text=True, timeout=10)
    fields = result.stdout.strip().split()
    ok = result.returncode == 0 and len(fields) == 2 and fields[0] != "000" and float(fields[1]) <= 1
    print("vinmec", result.stdout.strip(), "ok", ok, flush=True)
    return ok


def guard_probe():
    if not probe():
        time.sleep(30)
        if not probe():
            raise RuntimeError("vinmec repeated >1s/failure; stopping cooperatively")


def gpu_run(script, *args):
    """Run in-process; model hooks abort between forwards on repeated slow probes. No process kill."""
    import transformers
    guard_probe()
    last = [time.monotonic()]

    def before_forward(_model, _inputs):
        if time.monotonic() - last[0] >= 30:
            guard_probe()
            last[0] = time.monotonic()

    classes = [transformers.AutoModel, transformers.AutoModelForSequenceClassification]
    originals = [cls.from_pretrained for cls in classes]
    for cls, original in zip(classes, originals):
        def load(*a, _original=original, **kw):
            model = _original(*a, **kw)
            model.register_forward_pre_hook(before_forward)
            return model
        cls.from_pretrained = load
    argv = sys.argv
    try:
        sys.argv = [str(S / script), *map(str, args)]
        try:
            runpy.run_path(str(S / script), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        sys.argv = argv
        for cls, original in zip(classes, originals):
            cls.from_pretrained = original
    guard_probe()


def stage_embed():
    docs = G / "docs_embed"
    docs.mkdir(exist_ok=True)
    src = G / "docs_vi24_embed.jsonl.zst"
    if src.exists():
        shutil.move(str(src), str(docs / src.name))
    (G / "empty_emb").mkdir(exist_ok=True)
    gpu_run("embed_all.py", "--docs", docs, "--work-old", G / "empty_emb", "--work", G / "emb",
            "--queries", ROOT / "work/queries.jsonl", "--out", G / "dense_new_top200.json",
            "--doc-len", 512, "--query-len", 256, "--batch", 16, "--top", 200)
    gpu_run("dense_topk.py", "--emb", G / "emb", "--docs", docs / "docs_vi24_embed.jsonl.zst",
            "--queries", ROOT / "work/hyde/q_hyde.jsonl", "--query-len", 512, "--top", 200,
            "--device", "cuda", "--out", G / "hyde_new_top200.json")


def stage_pool():
    for kind in ["dense", "hyde"]:
        vi("dense-merge", "--cache-old", B / f"{kind}_v23_top200.json",
           "--cache-new", W / f"{kind}_new_top200.json", "--top", 200,
           "--out", W / f"{kind}_v24_top200.json", "--report", W / f"{kind}_merge_report.json")
    vi("pool-new", "--pool", B / "cache_pool_v23.json", "--bm25-new", W / "cache_vi24_k300.json",
       "--dense-new", W / "dense_v24_top200.json", "--new-ids", W / "new_ids.json",
       "--idx", W / "new_idx.json", "--out-cands", W / "cands_rr_new.json",
       "--out-new-hits", W / "new_hits.json", "--report", W / "pool_new_report.json")


def stage_rr():
    gpu_run("rerank_run.py", "--docs", G / "docs_embed/docs_vi24_embed.jsonl.zst",
            "--queries", ROOT / "work/queries.jsonl", "--cands", G / "cands_rr_new.json",
            "--out", G / "rerank_new.json", "--max-len", 512, "--batch", 16, "--top", 500,
            "--resume-dir", G / "rr_parts")


def missing_docwin(doclist, old_chunks, old_doclist, idx):
    """Reuse every vi23 candidate, including known empty-text pairs (no chunks)."""
    out = {}
    for q, docs in doclist.items():
        covered = {str(r[0]) for r in old_chunks.get(q, [])} | {str(d) for d in old_doclist.get(q, [])}
        rows = [[d, idx[str(d)]] for d in docs if str(d) not in covered]
        if rows:
            out[q] = rows
    return out


def stage_doclist():
    import numpy as np
    vi("pool-merge", "--pool-old", B / "cache_pool_v23.json", "--new-hits", W / "new_hits.json",
       "--rerank", W / "rerank_new.json", "--out", W / "cache_pool_v24.json", "--report", W / "pool_merge_report.json")
    assert read(W / "pool_merge_report.json")["hit_logit_None"] == 0
    vi("fuse", "--pool", W / "cache_pool_v24.json", "--q8b", Q8, "--dense", W / "dense_v24_top200.json",
       "--hyde", W / "hyde_v24_top200.json", "--out-rrf3d2", W / "cache_rrf3d2_v24.json",
       "--out-fhyde", W / "cache_fhyde_v24.json", "--report", W / "fuse_report.json")
    vi("doclist", "--pool", W / "cache_pool_v24.json", "--fhyde", W / "cache_fhyde_v24.json",
       "--ref-chunks", REF_C, "--out", W / "doclist_v24.json", "--report", W / "doclist_report.json")
    dl = read(W / "doclist_v24.json")
    idx = {str(d): i for i, d in enumerate(np.load(IX / "ids.npy"))}
    cands = missing_docwin(dl, read(B / "chunk_dw_u900_v23.json.zst"), read(B / "doclist_v23.json"), idx)
    new_ids = set(map(str, read(W / "new_ids.json")))
    # Previously scored p3 pairs can be reused even when absent from vi23's doclist.
    p3 = read(ROOT / "data/sub/diag18/docwin_p3/docwin_p3.json.zst")
    reuse, fresh = {}, {}
    for q, rows in cands.items():
        for h in rows:
            target = reuse if str(h[0]) in p3.get(q, {}) else fresh
            target.setdefault(q, []).append(h)
    write(W / "cands_docwin_vi24.json", fresh)
    write(W / "cands_docwin_reuse.json", reuse)
    write(W / "cache_docwin_old.json", {q: [[*h, 0.0] for h in rows if str(h[0]) not in new_ids]
                                           for q, rows in fresh.items()})
    run("export_docs.py", "--index", BASE_IX, "--cache", W / "cache_docwin_old.json",
        "--n", 1000, "--chars", 200000, "--no-title", "--part-mb", 80,
        "--out-docs", W / "docs_vi24_old.jsonl.zst", "--out-cands", W / "export_old_cands.json")
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
    docs = [G / "docs_vi24_docwin.jsonl.zst"] + sorted(G.glob("docs_vi24_old*.jsonl.zst"))
    gpu_run("docwin_run.py", "--docs", *docs, "--queries", ROOT / "work/queries.jsonl",
            "--cands", G / "cands_docwin_vi24.json", "--out", G / "docwin_vi24.json.zst",
            "--max-len", 512, "--batch", 16, "--top-docs", 500, "--window-words", 300,
            "--window-stride", 250, "--resume-dir", G / "dw_parts")


def merge_chunks(old, new):
    from dwx_merge import sort_rows
    out = {}
    for q in list(old) + [q for q in new if q not in old]:
        old_keys = {(str(r[0]), r[2]) for r in old.get(q, [])}
        assert not any((str(r[0]), r[2]) in old_keys for r in new.get(q, [])), "overlapping chunk pair"
        out[q] = sort_rows(old.get(q, []) + new.get(q, []))
    return out


def stage_chunk():
    from chunks_from_docwin import build_q_rows, select_q_rows, verify_verbatim
    from dwx_merge import write_zst_stream
    from rerank_run import load_docs
    cands, reuse = read(W / "cands_docwin_vi24.json"), read(W / "cands_docwin_reuse.json")
    wins = read(W / "docwin_vi24.json.zst")
    p3 = read(ROOT / "data/sub/diag18/docwin_p3/docwin_p3.json.zst")
    for q, rows in reuse.items():
        cands.setdefault(q, []).extend(rows)
        wins.setdefault(q, {}).update({str(h[0]): p3[q][str(h[0])] for h in rows})
    paths = [W / "docs_vi24_docwin.jsonl.zst", *sorted(W.glob("docs_vi24_old*.jsonl.zst")),
             ROOT / "data/sub/diag18/docwin_p3/docwin_p3.part000.jsonl.zst",
             ROOT / "data/sub/diag18/docwin_p3/docwin_p3.part001.jsonl.zst"]
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
    old = read(B / "chunk_dw_u900_v23.json.zst")
    merged = merge_chunks(old, new)
    write_zst_stream(W / "chunk_dw_u900_v24.json.zst", merged)
    write(W / "chunk_report.json", {"new_chunks": sum(map(len, new.values())), "verbatim_bad": bad,
          "empty_text_pairs": sum(not docs[h[0]].strip() for rows in cands.values() for h in rows),
          "old_chunks": sum(map(len, old.values())), "total_chunks": sum(map(len, merged.values()))})


def stage_build():
    delta = 48
    while True:
        vi(*chain_args(W, W / "chunk_dw_u900_v24.json.zst", "vi24", delta=delta))
        run("evaluate.py", "validate", W / "submission_vi24.json")
        run("zip_submission.py", "--json", W / "submission_vi24.json", "--zip", W / "submission_vi24.zip", "--method", "lzma")
        code = core.check_zip(str(W / "submission_vi24.json"), str(W / "submission_vi24.zip"),
                              str(W / "report_chain_vi24.json"), "vi24")
        if code == 0:
            break
        assert code == 3
        next_delta = read(W / "report_chain_vi24.json")["zip"]["vi24"]["next_delta"]
        assert 0 <= next_delta < delta
        delta = next_delta
    stage_stats()


def stage_stats():
    vi("stats", "--submission", W / "submission_vi24.json", "--fhyde", W / "cache_fhyde_v24.json",
       "--fhyde-old", B / "cache_fhyde_v23.json", "--pool", W / "cache_pool_v24.json",
       "--new-ids", W / "new_ids.json", "--links", W / "new_host.json", "--report", W / "stats_report.json")
    vi("newonly30", "--fhyde", W / "cache_fhyde_v24.json", "--new-ids", W / "new_ids.json",
       "--out", W / "submission_vi24_newonly30.json", "--report", W / "newonly30_report.json")
    run("evaluate.py", "validate", W / "submission_vi24_newonly30.json")
    run("zip_submission.py", "--json", W / "submission_vi24_newonly30.json",
        "--zip", W / "submission_vi24_newonly30.zip", "--method", "lzma")
    write(W / "zip_fingerprints.json", [fingerprint(W / n) for n in ["submission_vi24.zip", "submission_vi24_newonly30.zip"]])


def status(machine):
    work = G if machine == "z87" else W
    print("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    for p in sorted(work.glob("*.log")):
        with p.open("rb") as f:
            f.seek(max(0, p.stat().st_size - 5000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        print(p.name, "mtime", time.ctime(p.stat().st_mtime), "\n", "\n".join(lines[-8:]))
    print("locks", {p.name: p.read_text(errors="replace") for p in (ROOT / "locks").glob("*.lock")})
    tasks = subprocess.run(["schtasks", "/Query", "/FO", "CSV", "/NH"], capture_output=True, text=True)
    print("tasks", "\n".join(l for l in tasks.stdout.splitlines() if "vi24" in l))
    print("files", json.dumps({p.name: p.stat().st_size for p in work.glob("*") if p.is_file()}))


def reports(machine):
    work = G if machine == "z87" else W
    for name in ["prep_report.json", "verify_idx_report.json", "report_chain_km.json",
                 "pool_new_report.json", "pool_merge_report.json", "doclist_report.json",
                 "docwin_cands_report.json", "chunk_report.json", "report_chain_vi24.json",
                 "stats_report.json", "newonly30_report.json", "zip_fingerprints.json", "wincheck.json"]:
        p = work / name
        if p.exists():
            obj = read(p)
            if isinstance(obj, dict):
                obj = {k: v for k, v in obj.items() if k not in ["hist_kcap", "overlap_ids", "empty_ids"]}
            print(name, json.dumps(obj, ensure_ascii=False), flush=True)
    if machine == "b660" and (IX / "meta.json").exists():
        meta = read(IX / "meta.json")
        print("index", json.dumps({k: v for k, v in meta.items() if k != "shards"}))
        print("index_bytes", sum(p.stat().st_size for p in IX.iterdir() if p.is_file()))
        if (W / "cache_pool_v24.json").exists():
            from union_sub import adaptive_n
            for label, path in [("vi23", B / "cache_pool_v23.json"), ("vi24", W / "cache_pool_v24.json")]:
                pool = read(path)
                print("pool", label, "mean", sum(map(len, pool.values())) / len(pool),
                      "n_q_mean", sum(adaptive_n(h, -2.0, 40, 400) for h in pool.values()) / len(pool))


def cleanup(machine):
    lock = ROOT / "locks" / ("gpu.lock" if machine == "z87" else "heavy_job.lock")
    if lock.exists() and "vi24" in lock.read_text(errors="replace"):
        raise RuntimeError("vi24 lock still exists; inspect job before cleanup")
    stages = ["embed", "rr", "docwin"] if machine == "z87" else ["prep", "index", "sanity", "bm25", "pool", "doclist", "chunk", "build"]
    for stage in stages:
        name = "r2ai_vi24_" + stage
        query = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "CSV", "/NH"],
                               capture_output=True, text=True)
        if query.returncode:
            continue
        assert '"Running"' not in query.stdout, name + " still running"
        subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], check=True)
    print("CLEANUP DONE", machine, "locks", {p.name: p.read_text(errors="replace") for p in (ROOT / "locks").glob("*.lock")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prep", "sanity", "index", "exports", "bm25", "embed", "pool", "rr", "doclist", "docwin", "chunk", "build", "stats", "status", "reports", "cleanup"])
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
    with owned_lock(lock, f"vi24 {args.stage} pid={os.getpid()}"):
        globals()["stage_" + args.stage]()
    print("DONE", args.stage, flush=True)


if __name__ == "__main__":
    main()
