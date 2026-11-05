r"""vi36 = vi34 + goi handoff_vi_topup_20261026 (1.793 doc VI; thay doi DU LIEU; cong thuc chain giu y vi34).

Goi topup26 (1 file vi-topup-00000.jsonl.zst, 1.793 rec, sha 6bd8f4f2...) la bo sung cho goi full25
(id disjoint voi chi muc vi30 va moi shard recov; host laodong/qdnd/vov). Tat ca 1.793 id la MOI
(khong co trong chi muc vi30, khong thuoc bad 16.079) -> add het, khong id nao can crawl lai.

Cong thuc chain trung vi34 (KHONG doi tham so, chi them du lieu):
  fhyde w_q8b = 2 (rrf3d2/zd chunk = 1), n_q = so doc logit bge > -2.0 kep [40,400] (KHONG NQ_FUSED),
  U900, s = zc + 4*zd, b_cov 1, chain vi29.chain_build_v29 dedup doc_text, tap loai = bad chua crawl
  lai U ngoai kho, delta tu 56, tran zip LZMA 99.000.000 B.

Nen = vi34 (cac dau vao chain hieu luc cua vi34):
  pool = diag30/cache_pool_v30, q8b = diag33/q8b_v33, dense = diag30/dense_v30_top200,
  hyde = diag30/hyde_v30_top200, cands = vi34/chunk_dw_u900_v34, ref_chunks = diag29/in1/ref_chunks_v29.
Them 1.793 doc -> dense/hyde/pool/chunk gop them lop moi; q8b bu Modal (q8b_v36 = q8b_v33 + moi).

Hai luot fuse/doclist (theo PIPELINE_v4 3.7 + chi dan):
  luot 1 (q8b_v33): fuse + doclist -> doclist_pre; chon cap thieu 8B (doclist_pre + 30 hang bien) -> Modal.
  gop q8b_v36; luot 2 (q8b_v36): fuse + doclist -> doclist_v36 + cap docwin moi.

Stage b660 (deploy/b660_vi36.bat, D:\r2ai\data\sub\vi36\): prep sanity index bm25 pool doclist_pre
  q8b_select q8b_merge doclist chunk build stats. Stage z87 (deploy/z87_vi36.bat, D:\r2ai\work\vi36\):
  embed rr docwin. 8B Modal: deploy/modal_q8b36.py + scripts/q8b36.py (local).
Moi stage resume duoc. Khong nop bai; goi crawl CHI DOC; khong sua vi23/vi29/vi30/vi31/vi33/vi34/vi35.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi23 as core
import vi29 as v29
import vi30
import vi31
import vi33
from vi24 import (fingerprint, read, write, run, vi, owned_lock, gpu_run,
                  missing_docwin, merge_chunks)

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
# nen vi34 - CHI DOC
OUT30 = ROOT / "data/sub/diag30"
IN29 = ROOT / "data/sub/diag29/in1"
OUT34 = ROOT / "data/sub/vi34"
OUT33 = ROOT / "data/sub/vi33"
Q8_V33 = ROOT / "data/sub/diag33/q8b_v33.json"
CHUNK_V34 = OUT34 / "chunk_dw_u900_v34.json.zst"
IX30 = ROOT / "data/index/vi30"
P3 = ROOT / "data/sub/diag18/docwin_p3"
LINKS = vi30.LINKS
BAD_TSV = vi30.BAD_TSV
N_BAD = 16079
# goi moi - CHI DOC
PKG = ROOT / "data/crawl/handoff_vi_topup_20261026"
PKG_FILES = ["vi-topup-00000.jsonl.zst"]
N_PKG = 1793
PKG_SHA = "6bd8f4f2ca3b3cbf029b56b1e9b9e4248f36a15125f46e061177a0bf1815ae78"
# out
OUT = ROOT / "data/sub/vi36"
IN36 = OUT / "in1"
IN_S = OUT / "in_sanity"
IX = ROOT / "data/index/vi36"
G = ROOT / "work/vi36"
NAME, SANITY_NAME, PROBE = "vi36", "vi36_sanity", "vi36p"
SANITY_DELTA = 56       # delta cua submission_vi34
DELTA0 = 56
MAX_ZIP = 99_000_000
W8F = 2.0               # trong so 8B trong fhyde (giu y vi34)
EXP_ADD = 1793
WIN_N, WIN_TOL = 300, 0.02
IDENT_MAX = 0.02
KEEP_FLOOR = 0.95
FILE_NAMES = {"pool": "pool_v36.json", "q8b": "q8b_v36.json", "dense": "dense_v36.json",
              "hyde": "hyde_v36.json", "cands": "cands_v36.json.zst",
              "ref_chunks": "ref_chunks_v36.json"}


# ---------------------------------------------------------------- ham thuan (co test)

def sha_text(text) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def select_new_ids(pkg_ids, idx_ids) -> dict:
    """Id can them = id trong goi ma khong co trong chi muc cu. Goi nay disjoint -> add = tat ca.
    Tra {"add": set, "dup": set (trung chi muc cu, phai = rong)}."""
    ps, is_ = set(pkg_ids), set(idx_ids)
    return {"add": ps - is_, "dup": ps & is_}


def verify_index_ids(new_ids, old_ids, added) -> dict:
    """Tap id chi muc moi phai = old U added, khong trung/thieu/thua (vi30.verify_index_ids)."""
    return vi30.verify_index_ids(new_ids, old_ids, set(), added)


def filter_rows(rows: list, drop) -> list:
    """Ban ghi bo id thuoc drop (giu nguyen thu tu) - vi30.filter_rows."""
    return vi30.filter_rows(rows, drop)


def build_excl(bad, added) -> set:
    """Tap loai khi build: bad chua duoc them/crawl lai - vi30.build_excl."""
    return vi30.build_excl(bad, added)


def merge_topk_hits(base_hits: list, new_hits: list, top: int = 200) -> tuple:
    """Gop top-k dense/hyde: hit cu giu diem, trung lay cu (vi23.merge_topk)."""
    return core.merge_topk(base_hits, new_hits, top)


def select_8b_targets(doclist: dict, fhyde: dict, q8b: dict, margin: int = 30) -> dict:
    """{qid: [doc_id...]} can cham 8B = (doclist + `margin` hang bien theo fhyde) - da co diem q8b.
    Giong q8b33 (doclist + margin) nhung gom ca doclist khong nam trong top fhyde (ref-chunks)."""
    out = {}
    for q, docs in doclist.items():
        q = str(q)
        scored = {str(h[0]) for h in q8b.get(q, [])}
        seen, keep = set(), []
        for d in docs:
            s = str(d)
            if s not in scored and s not in seen:
                seen.add(s)
                keep.append(int(d))
        fh = fhyde.get(q, [])
        k = len(docs) + margin
        for h in fh[:k]:
            s = str(h[0])
            if s not in scored and s not in seen:
                seen.add(s)
                keep.append(int(h[0]))
        if keep:
            out[q] = keep
    return out


def check_wincheck(rep: dict) -> None:
    """Dung neu so cap != 300 hoac lech max > 0,02 (vi30.check_wincheck)."""
    vi30.check_wincheck(rep)


# ---------------------------------------------------------------- tien ich

def load_bad() -> set:
    return vi30.load_bad()


def load_excl_corpus():
    """excl = bad chua crawl lai (bad - id them vi30), corpus = tap VI chuan."""
    import vi25
    excl = vi30.build_excl(vi30.load_bad(), read(OUT30 / "new_ids.json"))
    return excl, v29.load_corpus_ids(LINKS)


def input_spec(q8b_src, cands_src) -> list:
    """6 dau vao chain vi36: pool/dense/hyde cua vi36 (da gop lop moi), q8b/cands tham so,
    ref_chunks giu nguyen."""
    return [(FILE_NAMES["pool"], OUT / "cache_pool_v36.json", "cache"),
            (FILE_NAMES["q8b"], Path(q8b_src), "cache"),
            (FILE_NAMES["dense"], OUT / "dense_v36_top200.json", "cache"),
            (FILE_NAMES["hyde"], OUT / "hyde_v36_top200.json", "cache"),
            (FILE_NAMES["cands"], Path(cands_src), "cands"),
            (FILE_NAMES["ref_chunks"], IN29 / "ref_chunks_v29.json", "ref")]


def sanity_spec() -> list:
    """Dau vao chain cho sanity tap-moi-rong = dau vao nen vi34 (pool/q8b/dense/hyde/cands/ref)."""
    return [(FILE_NAMES["pool"], OUT30 / "cache_pool_v30.json", "cache"),
            (FILE_NAMES["q8b"], Q8_V33, "cache"),
            (FILE_NAMES["dense"], OUT30 / "dense_v30_top200.json", "cache"),
            (FILE_NAMES["hyde"], OUT30 / "hyde_v30_top200.json", "cache"),
            (FILE_NAMES["cands"], CHUNK_V34, "cands"),
            (FILE_NAMES["ref_chunks"], IN29 / "ref_chunks_v29.json", "ref")]


def write_inputs_vi36(spec: list, indir: Path, excl: set, corpus) -> dict:
    """Ghi 6 dau vao chain da loc (bad + ngoai kho) vao indir - vi31.write_inputs_v31."""
    return vi31.write_inputs_v31(spec, indir, excl, corpus)


# ---------------------------------------------------------------- b660: prep / sanity / index

def stage_prep():
    import numpy as np
    from vi25 import corpus_vi_ids
    if (OUT / "prep_report.json").exists():
        print("prep da xong (resume): bo qua", flush=True)
        return
    t0 = time.time()
    man = core.check_manifest(PKG)
    write(OUT / "manifest_report.json", man)
    assert man["tat_ca_ok"] and man["file"] == len(PKG_FILES), man
    got = fingerprint(PKG / PKG_FILES[0])["sha256"]
    assert got == PKG_SHA, (got, PKG_SHA)
    rows = []
    for name in PKG_FILES:
        rows.extend(core.read_shard_records(PKG / name))
    assert len(rows) == N_PKG, len(rows)
    ids30 = [int(x) for x in np.load(IX30 / "ids.npy")]
    pkg_ids = {int(r["id"]) for r in rows if r.get("id") is not None}
    sel = select_new_ids(pkg_ids, set(ids30))
    so = {"add": len(sel["add"]), "dup": len(sel["dup"])}
    print("CHON", json.dumps(so), flush=True)
    if so["add"] != EXP_ADD or so["dup"] != 0:
        write(OUT / "prep_stop.json", {"ly_do": "so id add khac 1.793 hoac co id trung chi muc vi30", **so,
                                       "dup_vi_du": sorted(sel["dup"])[:20]})
        raise SystemExit(2)
    vi_ids, _all_ids = corpus_vi_ids(LINKS)
    cand_rows = [r for r in rows if r.get("id") is not None and int(r["id"]) in sel["add"]]
    kept, fstats, _ = core.filter_records(cand_rows, set())
    ngoai_vi = sorted(int(r["id"]) for r in kept if int(r["id"]) not in vi_ids)
    kept = [r for r in kept if int(r["id"]) in vi_ids]
    kept_ids = {int(r["id"]) for r in kept}
    if len(kept) < KEEP_FLOOR * EXP_ADD:
        write(OUT / "prep_stop.json", {"ly_do": "giu sau loc < 95%", "giu": len(kept),
                                       "filter": fstats, "ngoai_tap_vi": len(ngoai_vi)})
        raise SystemExit(2)
    bp30 = read(IX30 / "boilerplate.json")
    bp, bp_rep = core.fit_new_hosts(bp30, kept)
    IX.mkdir(parents=True, exist_ok=True)
    write(IX / "boilerplate.json", bp)
    from boilerplate import BoilerplateFilter
    filt = BoilerplateFilter.from_json(json.dumps(bp))
    empty = sum(1 for r in kept if not filt.clean(r.get("host", ""), r.get("text") or "").strip())
    vi30.write_zst_lines(OUT / "indexed" / "vi36-new.jsonl.zst", kept)
    write(OUT / "new_ids.json", sorted(kept_ids))
    write(OUT / "new_host.json", {str(r["id"]): r.get("host", "") for r in kept})
    rep = {"goi": str(PKG), "ban_ghi_goi": len(rows), "id_goi": len(pkg_ids),
           "chon": so, "id_trung_chi_muc_vi30": so["dup"],
           "filter": fstats, "ngoai_tap_vi": len(ngoai_vi), "ngoai_tap_vi_ids": ngoai_vi[:50],
           "giu": len(kept), "empty_after_boilerplate": empty, "boilerplate": bp_rep,
           "chi_muc_vi30": len(ids30), "giay": round(time.time() - t0, 1)}
    write(OUT / "prep_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "ngoai_tap_vi_ids"},
                     ensure_ascii=False), flush=True)


def stage_sanity():
    sr = OUT / "sanity_report.json"
    if sr.exists() and read(sr).get("ok"):
        print("sanity da xong (resume): bo qua", flush=True)
        return
    excl, corpus = load_excl_corpus()
    loc = write_inputs_vi36(sanity_spec(), IN_S, excl, corpus)
    files = v29.chain_files(IN_S, FILE_NAMES)
    vi33.set_w8f(W8F)
    core.NQ_FUSED = None
    try:
        v29.run_chain(SANITY_NAME, SANITY_DELTA, files, "doc_text", OUT)
    finally:
        vi33.set_w8f(None)
    cmp_rep = core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"),
                                str(OUT34 / "submission_vi34.json"))
    cmp_rep["loc"] = loc
    write(sr, cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    v29.require_sanity(cmp_rep)
    (OUT / f"submission_{SANITY_NAME}.json").unlink()


def stage_index():
    import numpy as np
    from bigindex import build
    vr = OUT / "verify_idx_report.json"
    if not (vr.exists() and read(vr).get("ok")):
        meta30 = read(IX30 / "meta.json")
        paths = [Path(p) for p in meta30["shards"]]
        paths.append(OUT / "indexed" / "vi36-new.jsonl.zst")
        assert all(p.exists() for p in paths)
        build(paths, IX, lang=meta30["lang"], nbits=meta30["nbits"], workers=3,
              budget=20_000_000, log=lambda m: print(m, flush=True))
        new = np.load(IX / "ids.npy")
        added = read(OUT / "new_ids.json")
        rep = verify_index_ids(new, np.load(IX30 / "ids.npy"), added)
        pos = {int(d): i for i, d in enumerate(new)}
        write(OUT / "new_idx.json", {str(d): pos[d] for d in added if d in pos})
        rep["new_idx"] = sum(1 for d in added if d in pos)
        write(vr, rep)
        print(json.dumps(rep), flush=True)
        assert rep["ok"], rep
    stage_exports()


def stage_exports():
    for mode in ["embed", "docwin"]:
        out = OUT / f"docs_vi36_{mode}.jsonl.zst"
        if (OUT / f"export_{mode}_report.json").exists() and out.exists():
            continue
        vi("export-new", "--shards", OUT / "indexed", "--bp", IX / "boilerplate.json",
           "--idx", OUT / "new_idx.json", "--mode", mode, "--out", out,
           "--report", OUT / f"export_{mode}_report.json")


def stage_bm25():
    if not (OUT / "cache_vi36_k300.json").exists():
        run("search_cache.py", "--index", IX, "--queries", ROOT / "data/query.parquet",
            "--out", OUT / "cache_vi36_k300.json", "-k", 300)
    vi("bm25-diff", "--cache-old", OUT30 / "cache_vi30_k300.json",
       "--cache-new", OUT / "cache_vi36_k300.json",
       "--new-ids", OUT / "new_ids.json", "--report", OUT / "bm25_diff.json")


# ---------------------------------------------------------------- z87

def stage_embed():
    docs = G / "docs_embed"
    docs.mkdir(parents=True, exist_ok=True)
    src = G / "docs_vi36_embed.jsonl.zst"
    if src.exists() and not (docs / src.name).exists():
        shutil.move(str(src), str(docs / src.name))
    (G / "empty_emb").mkdir(exist_ok=True)
    gpu_run("embed_all.py", "--docs", docs, "--work-old", G / "empty_emb", "--work", G / "emb",
            "--queries", ROOT / "work/queries.jsonl", "--out", G / "dense_new_top200.json",
            "--doc-len", 512, "--query-len", 256, "--batch", 16, "--top", 200)
    gpu_run("dense_topk.py", "--emb", G / "emb", "--docs", docs / "docs_vi36_embed.jsonl.zst",
            "--queries", ROOT / "work/hyde/q_hyde.jsonl", "--query-len", 512, "--top", 200,
            "--device", "cuda", "--out", G / "hyde_new_top200.json")


def stage_rr():
    gpu_run("rerank_run.py", "--docs", G / "docs_embed/docs_vi36_embed.jsonl.zst",
            "--queries", ROOT / "work/queries.jsonl", "--cands", G / "cands_rr_new.json",
            "--out", G / "rerank_new.json", "--max-len", 512, "--batch", 16, "--top", 500,
            "--resume-dir", G / "rr_parts")


def stage_docwin():
    if not (G / "wincheck.json").exists():
        gpu_run("dwx_p3.py", "winsample", "--docwin", ROOT / "work/docwin_ext/docwin_ext.json.zst",
                "--docs", ROOT / "work/docwin_ext/docwin_ext.part000.jsonl.zst",
                ROOT / "work/docwin_ext/docwin_ext.part001.jsonl.zst", "--queries", ROOT / "work/queries.jsonl",
                "--out", G / "winsample.json", "--n", WIN_N)
        gpu_run("dwx_p3.py", "score", "--in", G / "winsample.json", "--out", G / "winscore.json",
                "--model", "BAAI/bge-reranker-v2-m3", "--max-len", 512, "--batch", 16)
        gpu_run("dwx_p3.py", "wincheck", "--ref", G / "winsample.json", "--got", G / "winscore.json",
                "--report", G / "wincheck.json", "--tol", WIN_TOL)
    check_wincheck(read(G / "wincheck.json"))
    docs = [G / "docs_vi36_docwin.jsonl.zst"] + sorted(G.glob("docs_vi36_old*.jsonl.zst"))
    gpu_run("docwin_run.py", "--docs", *docs, "--queries", ROOT / "work/queries.jsonl",
            "--cands", G / "cands_docwin_vi36.json", "--out", G / "docwin_vi36.json.zst",
            "--max-len", 512, "--batch", 16, "--top-docs", 500, "--window-words", 300,
            "--window-stride", 250, "--resume-dir", G / "dw_parts")


# ---------------------------------------------------------------- b660: pool / doclist / 8B

def stage_pool():
    reps = {}
    for kind, base in [("dense", OUT30 / "dense_v30_top200.json"), ("hyde", OUT30 / "hyde_v30_top200.json")]:
        vi("dense-merge", "--cache-old", base, "--cache-new", OUT / f"{kind}_new_top200.json",
           "--top", 200, "--out", OUT / f"{kind}_v36_top200.json", "--report", OUT / f"{kind}_merge_report.json")
        reps[kind] = read(OUT / f"{kind}_merge_report.json")
    vi("pool-new", "--pool", OUT30 / "cache_pool_v30.json", "--bm25-new", OUT / "cache_vi36_k300.json",
       "--dense-new", OUT / "dense_v36_top200.json", "--new-ids", OUT / "new_ids.json",
       "--idx", OUT / "new_idx.json", "--out-cands", OUT / "cands_rr_new.json",
       "--out-new-hits", OUT / "new_hits.json", "--report", OUT / "pool_new_report.json")
    write(OUT / "pool_stage_report.json", reps)
    print(json.dumps(reps), flush=True)


def _pool_merge():
    if (OUT / "cache_pool_v36.json").exists():
        return
    vi("pool-merge", "--pool-old", OUT30 / "cache_pool_v30.json", "--new-hits", OUT / "new_hits.json",
       "--rerank", OUT / "rerank_new.json", "--out", OUT / "cache_pool_v36.json",
       "--report", OUT / "pool_merge_report.json")
    assert read(OUT / "pool_merge_report.json")["hit_logit_None"] == 0


def _fuse_doclist(q8b_src: Path, tag: str) -> dict:
    """fuse (rrf3d2 + fhyde w8f=2) + doclist voi q8b_src; tag='' -> ten chuan (doclist_v36.json)."""
    sfx = f"_{tag}" if tag else ""
    rrf = OUT / f"cache_rrf3d2_v36{sfx}.json"
    fh = OUT / f"cache_fhyde_v36{sfx}.json"
    dl = OUT / f"doclist_v36{sfx}.json"
    vi("fuse", "--pool", OUT / "cache_pool_v36.json", "--q8b", q8b_src,
       "--dense", OUT / "dense_v36_top200.json", "--hyde", OUT / "hyde_v36_top200.json",
       "--out-rrf3d2", rrf, "--out-fhyde", fh,
       "--w-q8b-fhyde", W8F, "--report", OUT / f"fuse_report{sfx}.json")
    vi("doclist", "--pool", OUT / "cache_pool_v36.json", "--fhyde", fh,
       "--ref-chunks", IN29 / "ref_chunks_v29.json", "--out", dl,
       "--report", OUT / f"doclist_report{sfx}.json")
    return {"doclist": dl, "fhyde": fh}


def stage_doclist_pre():
    if (OUT / "doclist_v36_pre.json").exists():
        print("doclist_pre da xong (resume): bo qua", flush=True)
        return
    _pool_merge()
    _fuse_doclist(Q8_V33, "pre")


def stage_q8b_select():
    rep = OUT / "q8b36_select_report.json"
    if rep.exists():
        print("q8b_select da xong (resume): bo qua", flush=True)
        return
    run("q8b36.py", "select", "--outdir", OUT / "q8b36_in", "--shards", "4", "--bench", "100",
        "--index", IX)


def stage_q8b_merge():
    if (OUT / "q8b_v36.json").exists() and (OUT / "doclist_v36.json").exists():
        print("q8b_merge da xong (resume): bo qua", flush=True)
        return
    news = sorted((OUT / "q8b36_in").glob("q8b36_shard*_out.json"))
    news += sorted((OUT / "q8b36_in").glob("q8b36_bench_new.json"))
    assert news, "khong co ket qua 8B tu Modal (q8b36_shard*_out.json)"
    run("q8b36.py", "merge", "--old", Q8_V33,
        "--new", *[str(p) for p in news], "--pool", OUT / "cache_pool_v36.json",
        "--new-ids", OUT / "new_ids.json", "--out", OUT / "q8b_v36.json",
        "--report", OUT / "q8b36_merge_report.json", "--fhyde", OUT / "cache_fhyde_v36_pre.json",
        "--bench-old", OUT / "q8b36_in" / "q8b36_bench_old.json",
        "--bench-new", OUT / "q8b36_in" / "q8b36_bench_new.json")
    _fuse_doclist(OUT / "q8b_v36.json", "")


def stage_doclist():
    if (OUT / "docwin_cands_report.json").exists():
        print("doclist da xong (resume): bo qua", flush=True)
        return
    import numpy as np
    if not (OUT / "doclist_v36.json").exists():
        _fuse_doclist(OUT / "q8b_v36.json", "")
    dl36 = read(OUT / "doclist_v36.json")
    q8 = v29.filter_cache(read(OUT / "q8b_v36.json"), *load_excl_corpus())
    write(OUT / "q8b_v36_clean.json", q8)
    wins30 = read(OUT30 / "docwin_vi30.json.zst")
    covered = vi31.union_doclists(read(IN29.parent / "doclist_vi29.json"), read(OUT30 / "doclist_v30.json"),
                                  {q: list(v) for q, v in wins30.items()},
                                  read(OUT34 / "doclist_vi34.json"))
    del wins30
    idx = {str(d): i for i, d in enumerate(np.load(IX / "ids.npy"))}
    chunks = read(CHUNK_V34)
    cands = missing_docwin(dl36, chunks, covered, idx)
    del chunks
    new_ids = set(map(str, read(OUT / "new_ids.json")))
    no_reuse = new_ids | {str(i) for i in load_bad()}
    reuse, fresh = vi30.split_docwin_cands(cands, read(P3 / "docwin_p3.json.zst"), no_reuse)
    write(OUT / "cands_docwin_vi36.json", fresh)
    write(OUT / "cands_docwin_reuse.json", reuse)
    rows = {q: [[*h, 0.0] for h in rs] for q, rs in fresh.items()}
    write(OUT / "cache_docwin_old.json", rows)
    if rows:
        run("export_docs.py", "--index", IX, "--cache", OUT / "cache_docwin_old.json",
            "--n", 1000, "--chars", 200000, "--no-title", "--part-mb", 80,
            "--out-docs", OUT / "docs_vi36_old.jsonl.zst", "--out-cands", OUT / "export_old_cands.json")
    n_fresh = sum(map(len, fresh.values()))
    n_new = sum(str(h[0]) in new_ids for rs in fresh.values() for h in rs)
    rep = {"doclist_vs_vi34": vi31.doclist_diff(dl36, read(OUT34 / "doclist_vi34.json")),
           "doclist_vs_vi34_top100": vi31.doclist_diff(dl36, read(OUT34 / "doclist_vi34.json"), 100),
           "phu_8b_doclist_v36": vi33.coverage_doclist(dl36, q8),
           "cap_chua_docwin": sum(map(len, cands.values())),
           "pairs": n_fresh, "new_doc_pairs": n_new, "old_doc_pairs": n_fresh - n_new,
           "p3_reused_pairs": sum(map(len, reuse.values())), "cau_co_cap_moi": len(fresh)}
    write(OUT / "docwin_cands_report.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- b660: chunk / build / stats

def stage_chunk():
    from chunks_from_docwin import build_q_rows, select_q_rows, verify_verbatim
    from dwx_merge import write_zst_stream
    from rerank_run import load_docs
    cands, reuse = read(OUT / "cands_docwin_vi36.json"), read(OUT / "cands_docwin_reuse.json")
    if not cands and not reuse:
        rep = {"new_chunks": 0, "ghep": False, "ghi_chu": "khong co cap moi: dung chunk_dw_u900_v34"}
        write(OUT / "chunk_report.json", rep)
        print(json.dumps(rep), flush=True)
        return
    wins = read(OUT / "docwin_vi36.json.zst") if cands else {}
    p3 = read(P3 / "docwin_p3.json.zst")
    for q, rows in reuse.items():
        cands.setdefault(q, []).extend(rows)
        wins.setdefault(q, {}).update({str(h[0]): p3[q][str(h[0])] for h in rows})
    del p3
    paths = [OUT / "docs_vi36_docwin.jsonl.zst", *sorted(OUT.glob("docs_vi36_old*.jsonl.zst")),
             P3 / "docwin_p3.part000.jsonl.zst", P3 / "docwin_p3.part001.jsonl.zst"]
    wanted = {h[0] for rows in cands.values() for h in rows}
    docs = load_docs([str(p) for p in paths if p.exists()], wanted=wanted)
    missing = wanted - set(docs)
    assert not missing, f"missing docwin texts: {len(missing)}"
    absent = [(q, h[0]) for q, rows in cands.items() for h in rows
              if str(h[0]) not in wins.get(q, {}) and docs[h[0]].strip()]
    assert not absent, f"missing nonempty docwin scores: {absent[:5]}"
    cache = {}
    new = {q: select_q_rows(build_q_rows(rows, docs, wins.get(q, {}), 900, 500, cache), 100000)
           for q, rows in cands.items()}
    nbad = verify_verbatim(new, docs)
    assert nbad == 0
    old = read(CHUNK_V34)
    merged = merge_chunks(old, new)
    write_zst_stream(OUT / "chunk_dw_u900_v36.json.zst", merged)
    rep = {"new_chunks": sum(map(len, new.values())), "ghep": True, "verbatim_bad": nbad,
           "empty_text_pairs": sum(not docs[h[0]].strip() for rows in cands.values() for h in rows),
           "old_chunks": sum(map(len, old.values())), "total_chunks": sum(map(len, merged.values()))}
    write(OUT / "chunk_report.json", rep)
    print(json.dumps(rep), flush=True)


def cands_source() -> Path:
    rep = read(OUT / "chunk_report.json") if (OUT / "chunk_report.json").exists() else {}
    return OUT / "chunk_dw_u900_v36.json.zst" if rep.get("new_chunks", 0) or rep.get("ghep") else CHUNK_V34


def stage_build():
    v29.OUT = OUT
    if (OUT / "build_report_vi36.json").exists() and (OUT / f"submission_{NAME}.zip").exists():
        print("build da xong (resume): bo qua", flush=True)
        return
    excl, corpus = load_excl_corpus()
    src = cands_source()
    loc = write_inputs_vi36(input_spec(OUT / "q8b_v36.json", src), IN36, excl, corpus)
    print(json.dumps(loc), flush=True)
    files = v29.chain_files(IN36, FILE_NAMES)
    vi33.set_w8f(W8F)
    core.NQ_FUSED = None
    try:
        delta, attempts = v29.find_delta(files, "doc_text", MAX_ZIP, DELTA0, PROBE)
        r = v29.chain_zip_check(NAME, delta, MAX_ZIP, files, "doc_text", validate=True)
    finally:
        vi33.set_w8f(None)
    assert r["code"] == 0, r
    clean = v29.assert_clean(read(OUT / f"submission_{NAME}.json"), excl, corpus)
    fp = fingerprint(OUT / f"submission_{NAME}.zip")
    write(OUT / "build_report_vi36.json",
          {"name": NAME, "delta": delta, "attempts": attempts, "max_zip": MAX_ZIP,
           "q8b": str(OUT / "q8b_v36.json"), "cands": str(src), "loc_dau_vao": loc,
           "assert_clean": clean, "validate": "ok", "dedup_mode": "doc_text", "zip": fp,
           "n_q_TB": r.get("n_q_TB"), "tong_chunk": r.get("tong_chunk")})
    print(json.dumps({"build": NAME, "delta": delta, "attempts": attempts, **clean,
                      "zip_bytes": fp["bytes"], "sha256": fp["sha256"]}), flush=True)


def stage_stats():
    sub = read(OUT / f"submission_{NAME}.json")
    rep36 = read(OUT / f"report_chain_{NAME}.json")
    r36 = v29.sub_row(sub, rep36, fingerprint(OUT / f"submission_{NAME}.zip"), rep36.get("delta"))
    dl36 = {str(it["id"]): [str(d) for d in it["relevant_docs"]] for it in sub}
    new_ids = set(map(str, read(OUT / "new_ids.json")))
    n = max(len(sub), 1)
    d_new = c_new = 0
    for it in sub:
        docs = [str(d) for d in it["relevant_docs"]]
        d_new += sum(d in new_ids for d in docs)
        c_new += sum(str(c["doc_id"]) in new_ids for c in it["relevant_chunks"])
    del sub
    ref = read(OUT34 / "submission_vi34.json")
    rep34 = read(OUT34 / "report_chain_vi34.json") if (OUT34 / "report_chain_vi34.json").exists() \
        else read(OUT34 / "report_chain_vi31.json")
    r34 = v29.sub_row(ref, rep34, fingerprint(OUT34 / "submission_vi34.zip"), rep34.get("delta"))
    doi = trung30 = 0
    for it in ref:
        a = dl36.get(str(it["id"]), [])
        b = [str(d) for d in it["relevant_docs"]]
        doi += a != b
        trung30 += len(set(a[:30]) & set(b[:30]))
    del ref
    rep = {"vi36": r36, "vi34_tham_chieu": r34,
           "doclist_vs_vi34": vi31.doclist_diff(dl36, {str(it["id"]): [str(d) for d in it["relevant_docs"]]
                                                      for it in read(OUT34 / "submission_vi34.json")}),
           "TB_doc_moi_cau": round(d_new / n, 2), "TB_chunk_tu_doc_moi_cau": round(c_new / n, 2),
           "so_cau_doclist_doi": doi, "TB_top30_trung_vi34": round(trung30 / n, 2)}
    write(OUT / "stats_report_vi36.json", rep)
    write(OUT / "zip_fingerprints.json", [fingerprint(OUT / f"submission_{NAME}.zip")])
    print(json.dumps(rep, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- status / cleanup / main

def status(machine):
    work = G if machine == "z87" else OUT
    print("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    for p in sorted(work.glob("*.log")):
        with p.open("rb") as f:
            f.seek(max(0, p.stat().st_size - 3000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        print(p.name, "mtime", time.ctime(p.stat().st_mtime), "\n", "\n".join(lines[-10:]))
    print("locks", {p.name: p.read_text(errors="replace") for p in (ROOT / "locks").glob("*.lock")})


def cleanup(machine):
    lock = ROOT / "locks" / ("gpu.lock" if machine == "z87" else "heavy_job.lock")
    if lock.exists() and "vi36" in lock.read_text(errors="replace"):
        raise RuntimeError("vi36 lock still exists; inspect job before cleanup")
    name = "r2ai_vi36_" + machine
    q = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "CSV", "/NH"], capture_output=True, text=True)
    if not q.returncode:
        assert '"Running"' not in q.stdout, name + " still running"
        subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], check=True)
    print("CLEANUP DONE", machine, {p.name: p.read_text(errors="replace") for p in (ROOT / "locks").glob("*.lock")})


B660_STAGES = ["prep", "sanity", "index", "bm25", "pool", "doclist_pre", "q8b_select",
               "q8b_merge", "doclist", "chunk", "build", "stats"]
Z87_STAGES = ["embed", "rr", "docwin"]
# q8b_select chay q8b31.cmd_select (subprocess) tu giu heavy_job.lock -> vi36 main KHONG khoa.
NO_SELF_LOCK = {"q8b_select"}


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=B660_STAGES + Z87_STAGES + ["status", "cleanup"])
    parser.add_argument("--machine", choices=["b660", "z87"], default="b660")
    args = parser.parse_args()
    if args.stage == "status":
        status(args.machine)
        return
    if args.stage == "cleanup":
        cleanup(args.machine)
        return
    work = G if args.machine == "z87" else OUT
    work.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "locks" / ("gpu.lock" if args.machine == "z87" else "heavy_job.lock")
    if args.stage in NO_SELF_LOCK:
        globals()["stage_" + args.stage]()
    else:
        with owned_lock(lock, f"vi36 {args.stage} pid={os.getpid()}"):
            globals()["stage_" + args.stage]()
    print("DONE", args.stage, flush=True)


if __name__ == "__main__":
    main()
