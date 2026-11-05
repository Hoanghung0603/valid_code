r"""vi30 = vi29 + goi handoff_vi_20261025_full (thay doi DU LIEU; cong thuc chain giu y vi29).

Goi full25 (122.859 rec, supersede 3 goi _fix) so voi chi muc vi25 + 16.079 id nhiem:
  - id da co trong chi muc vi25 va khong thuoc bad (96.090, text giong het) -> bo qua;
  - id thuoc bad da crawl lai (12.705, chi muc vi25 dang giu text SAI cu) -> them lai voi text moi;
  - id chua co trong chi muc vi25 (14.064) -> them.
Moi du lieu suy tu text cu cua 16.079 id (embedding, logit bge/8B, docwin, chunk ung vien, cf4)
KHONG dung: nguon cu = cac file in1 cua vi29 (da loc bad + ngoai kho), phan moi tinh tren text moi.
Doc them (ke ca 12.705 id crawl lai) khong co diem 8B, nhu doc moi cua vi23-25.

Chi muc vi30 = shard cua chi muc vi25 (shard nao chua id bad thay bang ban loc bo bad) + shard
vi30-new; kiem tap id = (vi25 - bad) U id them. Tap loai khi build = bad chua crawl lai U ngoai kho.

Stage b660 (deploy/b660_vi30.bat, D:\r2ai\data\sub\diag30\): prep, sanity, index (+exports), bm25,
pool, doclist, chunk, build, stats. Stage z87 (deploy/z87_vi30.bat, D:\r2ai\work\vi30\): embed, rr,
docwin. Chuyen file giua hai may qua may local. Moi stage resume duoc (bo qua neu da co report).
Khong nop bai; artefact vi25/vi28/vi29 va goi crawl CHI DOC.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi23 as core
from vi24 import (fingerprint, read, write, run, vi, owned_lock, gpu_run, missing_docwin,
                  merge_chunks)

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
W25 = ROOT / "data/sub/diag25"                  # vi25 - CHI DOC
OUT29 = ROOT / "data/sub/diag29"                # vi29 - CHI DOC
IN29 = OUT29 / "in1"
OUT = ROOT / "data/sub/diag30"
IN30 = OUT / "in1"
IX25 = ROOT / "data/index/vi25"                 # CHI DOC
IX = ROOT / "data/index/vi30"
G = ROOT / "work/vi30"
PKG = ROOT / "data/crawl/handoff_vi_20261025_full"   # CHI DOC
PKG_FILES = [f"vi-recov-{i:05d}.jsonl.zst" for i in range(7)]
N_PKG = 122859
BAD_TSV = ROOT / "data/crawl/async_recovery_20261009/audit/contamination_bad_ids_20261024.tsv"
N_BAD = 16079
LINKS = ROOT / "data/sub/diag19/links_corpus_vi_en_zh.parquet"
P3 = ROOT / "data/sub/diag18/docwin_p3"
NAME, SANITY_NAME, PROBE = "vi30", "vi30_sanity", "vi30p"
SANITY_DELTA = 58   # delta cua submission_vi29
DELTA0 = 58
MAX_ZIP = 99_000_000
EXP_ADD = 26769
KEEP_FLOOR = 0.95
IDENT_MAX = 0.02
WIN_N, WIN_TOL = 300, 0.02
FP4 = (267640, 268801, 269213, 572834)
FILE_NAMES = {"pool": "pool_v30.json", "q8b": "q8b_v30.json", "dense": "dense_v30.json",
              "hyde": "hyde_v30.json", "cands": "cands_v30.json.zst",
              "ref_chunks": "ref_chunks_v30.json"}


# ---------------------------------------------------------------- ham thuan (co test)

def read_ids_tsv(path) -> list:
    """id (cot dau) cua TSV, bo header/dong rong (giong vi28.read_ids_tsv)."""
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines()):
        line = line.strip()
        if not line:
            continue
        first = line.split("\t")[0].strip()
        if not first.lstrip("-").isdigit():
            if i == 0 or not out:
                continue
            raise ValueError(f"dong khong phai id: {path} {line[:40]}")
        out.append(int(first))
    return out


def sha_text(text) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def select_add_ids(pkg_ids, idx_ids, bad) -> dict:
    """Id can them = id trong goi ma (khong co trong chi muc cu) hoac (thuoc bad).
    Id cu khong bad giu nguyen trong chi muc (skip)."""
    idx_s, bad_s, pkg_s = set(idx_ids), set(bad), set(pkg_ids)
    add = {i for i in pkg_s if i not in idx_s or i in bad_s}
    return {"add": add, "recraw": add & bad_s, "fresh": add - bad_s, "skip": pkg_s - add}


def filter_rows(rows: list, drop) -> list:
    """Ban ghi shard bo id thuoc drop (giu nguyen thu tu, ban ghi thieu id giu nguyen)."""
    d = set(drop)
    return [r for r in rows if r.get("id") is None or int(r["id"]) not in d]


def verify_index_ids(new_ids, old_ids, bad, added) -> dict:
    """Tap id chi muc moi phai = (cu - bad) U them, khong trung."""
    new = [int(x) for x in new_ids]
    s = set(new)
    want = ({int(x) for x in old_ids} - set(bad)) | {int(x) for x in added}
    rep = {"so_doc": len(new), "trung": len(new) - len(s), "thieu": len(want - s),
           "thua": len(s - want), "mong_doi": len(want)}
    rep["ok"] = rep["trung"] == 0 and rep["thieu"] == 0 and rep["thua"] == 0
    return rep


def drop_stale_hits(cache: dict, bad) -> dict:
    """{qid: hits} bo hit cua id bad (idempotent)."""
    bs = {str(i) for i in bad}
    return {str(q): [h for h in hits or [] if str(h[0]) not in bs] for q, hits in cache.items()}


def merge_dense_old_new(cache_old: dict, cache_new: dict, bad, top: int = 200) -> tuple:
    """Loc hit cu cua id bad (embedding text cu) roi gop top-k theo cosine (vi23.merge_topk)."""
    old = drop_stale_hits(cache_old, bad)
    new = {str(q): v for q, v in cache_new.items()}
    out, n_moi, n_bo = {}, 0, 0
    for q in sorted(set(old) | set(new), key=lambda x: int(x)):
        hits, st = core.merge_topk(old.get(q, []), new.get(q, []), top)
        out[q] = hits
        n_moi += st["so_moi"]
    for q, hits in cache_old.items():
        n_bo += len(hits or []) - len(old.get(str(q), []))
    n = max(len(out), 1)
    return out, {"so_cau": len(out), "top": top, "TB_doc_moi_lot": n_moi / n, "hit_cu_bad_bo": n_bo}


def split_docwin_cands(cands: dict, p3: dict, no_reuse) -> tuple:
    """Cap docwin con thieu -> (reuse tu p3, fresh). Doc trong no_reuse (doc them/bad: text
    cu hoac khong co) luon fresh."""
    nr = {str(i) for i in no_reuse}
    reuse, fresh = {}, {}
    for q, rows in cands.items():
        for h in rows:
            ok = str(h[0]) in p3.get(q, {}) and str(h[0]) not in nr
            (reuse if ok else fresh).setdefault(q, []).append(h)
    return reuse, fresh


def build_excl(bad, added) -> set:
    """Tap loai khi build: id bad chua crawl lai (bad - id them)."""
    return set(bad) - {int(x) for x in added}


def fp4_status(ids, sel: dict, kept_ids, idx_ids, bad, vi_ids) -> dict:
    out = {}
    for i in ids:
        out[str(i)] = {"trong_bad": i in bad, "trong_chi_muc_vi25": i in idx_ids,
                       "duoc_chon": i in sel["add"], "trong_tap_vi": i in vi_ids,
                       "giu_sau_loc": i in kept_ids,
                       "con_trong_chi_muc_vi30": (i in kept_ids) or (i in idx_ids and i not in bad)}
    return out


def write_zst_lines(path: Path, rows) -> int:
    import zstandard
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            for r in rows:
                z.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
                n += 1
    os.replace(tmp, path)
    return n


def load_bad() -> set:
    bad = set(read_ids_tsv(BAD_TSV))
    assert len(bad) == N_BAD, len(bad)
    return bad


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
    rows = []
    for name in PKG_FILES:
        rows.extend(core.read_shard_records(PKG / name))
    assert len(rows) == N_PKG, len(rows)
    bad = load_bad()
    ids25 = [int(x) for x in np.load(IX25 / "ids.npy")]
    ids25_s = set(ids25)
    part25 = np.load(IX25 / "doc_part.npy")
    meta25 = read(IX25 / "meta.json")
    pkg_ids = {int(r["id"]) for r in rows if r.get("id") is not None}
    sel = select_add_ids(pkg_ids, ids25_s, bad)
    so = {k: len(v) for k, v in sel.items()}
    print("CHON", json.dumps(so), flush=True)
    if so["add"] != EXP_ADD:
        write(OUT / "prep_stop.json", {"ly_do": "so id chon khac 26.769", **so})
        raise SystemExit(2)

    # shard cu chua id bad -> ban loc; dong thoi lay sha text cu cua id crawl lai
    flagged = sorted({int(part25[i]) for i, d in enumerate(ids25) if d in bad})
    recraw = sel["recraw"]
    old_sha, shard_loc = {}, {}
    for pi in flagged:
        src = Path(meta25["shards"][pi])
        srows = core.read_shard_records(src)
        for r in srows:
            d = r.get("id")
            if d is not None and int(d) in recraw and r.get("status") == 200 and int(d) not in old_sha:
                old_sha[int(d)] = sha_text(r.get("text"))
        kept_rows = filter_rows(srows, bad)
        dest = OUT / "shards_fix" / f"{src.parent.parent.name}_{src.name}"
        write_zst_lines(dest, kept_rows)
        shard_loc[str(src)] = {"dest": str(dest), "truoc": len(srows), "sau": len(kept_rows),
                               "bo": len(srows) - len(kept_rows)}
        print("SHARD", json.dumps(shard_loc[str(src)]), flush=True)
        del srows, kept_rows

    first = {}
    for r in rows:
        if r.get("id") is not None and int(r["id"]) not in first:
            first[int(r["id"])] = r
    giong = sorted(d for d in recraw if old_sha.get(d) == sha_text(first[d].get("text")))
    write(OUT / "recraw_oldsha.json", {str(d): {"old": old_sha.get(d), "new": sha_text(first[d].get("text"))}
                                       for d in sorted(recraw)})
    ty_le_giong = len(giong) / max(len(recraw), 1)
    if ty_le_giong > IDENT_MAX:
        write(OUT / "prep_stop.json", {"ly_do": "text crawl lai giong het > 2%", "giong": len(giong),
                                       "recraw": len(recraw), "vi_du": giong[:20]})
        raise SystemExit(2)

    vi_ids, all_ids = corpus_vi_ids(LINKS)
    cand_rows = [r for r in rows if r.get("id") is not None and int(r["id"]) in sel["add"]]
    kept, fstats, _ = core.filter_records(cand_rows, set())
    ngoai_vi = sorted(int(r["id"]) for r in kept if int(r["id"]) not in vi_ids)
    kept = [r for r in kept if int(r["id"]) in vi_ids]
    kept_ids = {int(r["id"]) for r in kept}
    if len(kept) < KEEP_FLOOR * EXP_ADD:
        write(OUT / "prep_stop.json", {"ly_do": "giu sau loc < 95%", "giu": len(kept),
                                       "filter": fstats, "ngoai_tap_vi": len(ngoai_vi)})
        raise SystemExit(2)
    bp, bp_rep = core.fit_new_hosts(read(IX25 / "boilerplate.json"), kept)
    IX.mkdir(parents=True, exist_ok=True)
    write(IX / "boilerplate.json", bp)
    from boilerplate import BoilerplateFilter
    filt = BoilerplateFilter.from_json(json.dumps(bp))
    empty = sum(1 for r in kept if not filt.clean(r.get("host", ""), r.get("text") or "").strip())
    write_zst_lines(OUT / "indexed" / "vi30-new.jsonl.zst", kept)
    write(OUT / "new_ids.json", sorted(kept_ids))
    write(OUT / "recraw_ids.json", sorted(kept_ids & bad))
    write(OUT / "new_host.json", {str(r["id"]): r.get("host", "") for r in kept})
    rep = {"goi": str(PKG), "ban_ghi_goi": len(rows), "id_goi": len(pkg_ids), "chon": so,
           "shard_cu_bi_loc": len(flagged), "shard_loc": shard_loc,
           "recraw_co_text_cu": len(old_sha), "recraw_text_giong_het": len(giong),
           "recraw_text_giong_het_ty_le": round(ty_le_giong, 5),
           "filter": fstats, "ngoai_tap_vi": len(ngoai_vi), "ngoai_tap_vi_ids": ngoai_vi[:50],
           "giu": len(kept), "giu_recraw": len(kept_ids & bad), "giu_moi": len(kept_ids - bad),
           "empty_after_boilerplate": empty, "boilerplate": bp_rep,
           "chi_muc_vi25_giu_lai": len(ids25_s - bad),
           "fp4": fp4_status(FP4, sel, kept_ids, ids25_s, bad, vi_ids),
           "giay": round(time.time() - t0, 1)}
    write(OUT / "prep_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("shard_loc", "ngoai_tap_vi_ids")},
                     ensure_ascii=False), flush=True)


def stage_sanity():
    import vi29 as v29
    if (OUT / "sanity_report.json").exists() and read(OUT / "sanity_report.json").get("ok"):
        print("sanity da xong (resume): bo qua", flush=True)
        return
    files = v29.chain_files(IN29, v29.FILE_NAMES)
    v29.run_chain(SANITY_NAME, SANITY_DELTA, files, "doc_text", OUT)
    cmp_rep = core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"),
                                str(OUT29 / "submission_vi29.json"))
    write(OUT / "sanity_report.json", cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    v29.require_sanity(cmp_rep)
    (OUT / f"submission_{SANITY_NAME}.json").unlink()


def stage_index():
    import numpy as np
    from bigindex import build
    vr = OUT / "verify_idx_report.json"
    if not (vr.exists() and read(vr).get("ok")):
        prep = read(OUT / "prep_report.json")
        meta25 = read(IX25 / "meta.json")
        loc = prep["shard_loc"]
        paths = [Path(loc[p]["dest"]) if p in loc else Path(p) for p in meta25["shards"]]
        paths.append(OUT / "indexed" / "vi30-new.jsonl.zst")
        assert all(p.exists() for p in paths)
        build(paths, IX, lang=meta25["lang"], nbits=meta25["nbits"], workers=3,
              budget=20_000_000, log=lambda m: print(m, flush=True))
        new = np.load(IX / "ids.npy")
        added = read(OUT / "new_ids.json")
        rep = verify_index_ids(new, np.load(IX25 / "ids.npy"), load_bad(), added)
        pos = {int(d): i for i, d in enumerate(new)}
        write(OUT / "new_idx.json", {str(d): pos[d] for d in added if d in pos})
        rep["new_idx"] = sum(1 for d in added if d in pos)
        write(vr, rep)
        print(json.dumps(rep), flush=True)
        assert rep["ok"], rep
    stage_exports()


def stage_exports():
    for mode in ["embed", "docwin"]:
        out = OUT / f"docs_vi30_{mode}.jsonl.zst"
        if (OUT / f"export_{mode}_report.json").exists() and out.exists():
            continue
        vi("export-new", "--shards", OUT / "indexed", "--bp", IX / "boilerplate.json",
           "--idx", OUT / "new_idx.json", "--mode", mode, "--out", out,
           "--report", OUT / f"export_{mode}_report.json")


def stage_bm25():
    if not (OUT / "cache_vi30_k300.json").exists():
        run("search_cache.py", "--index", IX, "--queries", ROOT / "data/query.parquet",
            "--out", OUT / "cache_vi30_k300.json", "-k", 300)
    vi("bm25-diff", "--cache-old", W25 / "cache_vi25_k300.json", "--cache-new", OUT / "cache_vi30_k300.json",
       "--new-ids", OUT / "new_ids.json", "--report", OUT / "bm25_diff.json")


# ---------------------------------------------------------------- z87

def stage_embed():
    docs = G / "docs_embed"
    docs.mkdir(exist_ok=True)
    src = G / "docs_vi30_embed.jsonl.zst"
    if src.exists() and not (docs / src.name).exists():
        shutil.move(str(src), str(docs / src.name))
    (G / "empty_emb").mkdir(exist_ok=True)
    gpu_run("embed_all.py", "--docs", docs, "--work-old", G / "empty_emb", "--work", G / "emb",
            "--queries", ROOT / "work/queries.jsonl", "--out", G / "dense_new_top200.json",
            "--doc-len", 512, "--query-len", 256, "--batch", 16, "--top", 200)
    gpu_run("dense_topk.py", "--emb", G / "emb", "--docs", docs / "docs_vi30_embed.jsonl.zst",
            "--queries", ROOT / "work/hyde/q_hyde.jsonl", "--query-len", 512, "--top", 200,
            "--device", "cuda", "--out", G / "hyde_new_top200.json")


def stage_rr():
    gpu_run("rerank_run.py", "--docs", G / "docs_embed/docs_vi30_embed.jsonl.zst",
            "--queries", ROOT / "work/queries.jsonl", "--cands", G / "cands_rr_new.json",
            "--out", G / "rerank_new.json", "--max-len", 512, "--batch", 16, "--top", 500,
            "--resume-dir", G / "rr_parts")


def check_wincheck(rep: dict) -> None:
    """Dung neu so cap != 300 hoac lech max > 0,02."""
    ok = rep.get("so_cap") == WIN_N and rep.get("so_mau") == WIN_N and \
        rep.get("max_sai") is not None and rep["max_sai"] <= WIN_TOL
    if not ok:
        print("WINCHECK FAIL", json.dumps(rep), flush=True)
        raise SystemExit(2)


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
    docs = [G / "docs_vi30_docwin.jsonl.zst"] + sorted(G.glob("docs_vi30_old*.jsonl.zst"))
    gpu_run("docwin_run.py", "--docs", *docs, "--queries", ROOT / "work/queries.jsonl",
            "--cands", G / "cands_docwin_vi30.json", "--out", G / "docwin_vi30.json.zst",
            "--max-len", 512, "--batch", 16, "--top-docs", 500, "--window-words", 300,
            "--window-stride", 250, "--resume-dir", G / "dw_parts")


# ---------------------------------------------------------------- b660: pool / doclist / chunk

def stage_pool():
    bad = load_bad()
    reps = {}
    for kind, old in [("dense", IN29 / "dense_v29.json"), ("hyde", IN29 / "hyde_v29.json")]:
        out, rep = merge_dense_old_new(core.load_cache(old), core.load_cache(OUT / f"{kind}_new_top200.json"),
                                       bad, 200)
        write(OUT / f"{kind}_v30_top200.json", out)
        reps[kind] = rep
    write(OUT / "dense_merge_report.json", reps)
    print(json.dumps(reps), flush=True)
    vi("pool-new", "--pool", IN29 / "pool_v29.json", "--bm25-new", OUT / "cache_vi30_k300.json",
       "--dense-new", OUT / "dense_v30_top200.json", "--new-ids", OUT / "new_ids.json",
       "--idx", OUT / "new_idx.json", "--out-cands", OUT / "cands_rr_new.json",
       "--out-new-hits", OUT / "new_hits.json", "--report", OUT / "pool_new_report.json")


def stage_doclist():
    import numpy as np
    vi("pool-merge", "--pool-old", IN29 / "pool_v29.json", "--new-hits", OUT / "new_hits.json",
       "--rerank", OUT / "rerank_new.json", "--out", OUT / "cache_pool_v30.json",
       "--report", OUT / "pool_merge_report.json")
    assert read(OUT / "pool_merge_report.json")["hit_logit_None"] == 0
    vi("fuse", "--pool", OUT / "cache_pool_v30.json", "--q8b", IN29 / "q8b_v29.json",
       "--dense", OUT / "dense_v30_top200.json", "--hyde", OUT / "hyde_v30_top200.json",
       "--out-rrf3d2", OUT / "cache_rrf3d2_v30.json", "--out-fhyde", OUT / "cache_fhyde_v30.json",
       "--report", OUT / "fuse_report.json")
    vi("doclist", "--pool", OUT / "cache_pool_v30.json", "--fhyde", OUT / "cache_fhyde_v30.json",
       "--ref-chunks", IN29 / "ref_chunks_v29.json", "--out", OUT / "doclist_v30.json",
       "--report", OUT / "doclist_report.json")
    dl = read(OUT / "doclist_v30.json")
    idx = {str(d): i for i, d in enumerate(np.load(IX / "ids.npy"))}
    cands = missing_docwin(dl, read(IN29 / "cands_v29.json.zst"), read(OUT29 / "doclist_vi29.json"), idx)
    new_ids = set(map(str, read(OUT / "new_ids.json")))
    no_reuse = new_ids | {str(i) for i in load_bad()}
    reuse, fresh = split_docwin_cands(cands, read(P3 / "docwin_p3.json.zst"), no_reuse)
    write(OUT / "cands_docwin_vi30.json", fresh)
    write(OUT / "cands_docwin_reuse.json", reuse)
    old_rows = {q: [[*h, 0.0] for h in rows if str(h[0]) not in new_ids] for q, rows in fresh.items()}
    old_rows = {q: v for q, v in old_rows.items() if v}
    write(OUT / "cache_docwin_old.json", old_rows)
    if old_rows:
        run("export_docs.py", "--index", IX, "--cache", OUT / "cache_docwin_old.json",
            "--n", 1000, "--chars", 200000, "--no-title", "--part-mb", 80,
            "--out-docs", OUT / "docs_vi30_old.jsonl.zst", "--out-cands", OUT / "export_old_cands.json")
    n_new = sum(str(h[0]) in new_ids for rows in fresh.values() for h in rows)
    rep = {"pairs": sum(map(len, fresh.values())), "new_doc_pairs": n_new,
           "old_doc_pairs": sum(map(len, fresh.values())) - n_new,
           "p3_reused_pairs": sum(map(len, reuse.values()))}
    write(OUT / "docwin_cands_report.json", rep)
    print(json.dumps(rep), flush=True)


def stage_chunk():
    from chunks_from_docwin import build_q_rows, select_q_rows, verify_verbatim
    from dwx_merge import write_zst_stream
    from rerank_run import load_docs
    cands, reuse = read(OUT / "cands_docwin_vi30.json"), read(OUT / "cands_docwin_reuse.json")
    wins = read(OUT / "docwin_vi30.json.zst")
    p3 = read(P3 / "docwin_p3.json.zst")
    for q, rows in reuse.items():
        cands.setdefault(q, []).extend(rows)
        wins.setdefault(q, {}).update({str(h[0]): p3[q][str(h[0])] for h in rows})
    del p3
    paths = [OUT / "docs_vi30_docwin.jsonl.zst", *sorted(OUT.glob("docs_vi30_old*.jsonl.zst")),
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
    nbad = verify_verbatim(new, docs)
    assert nbad == 0
    old = read(IN29 / "cands_v29.json.zst")
    bs = {str(i) for i in load_bad()}
    assert not any(str(r[0]) in bs for rows in old.values() for r in rows), "cands vi29 con id bad"
    merged = merge_chunks(old, new)
    write_zst_stream(OUT / "chunk_dw_u900_v30.json.zst", merged)
    rep = {"new_chunks": sum(map(len, new.values())), "verbatim_bad": nbad,
           "empty_text_pairs": sum(not docs[h[0]].strip() for rows in cands.values() for h in rows),
           "old_chunks": sum(map(len, old.values())), "total_chunks": sum(map(len, merged.values()))}
    write(OUT / "chunk_report.json", rep)
    print(json.dumps(rep), flush=True)


# ---------------------------------------------------------------- b660: build / stats

def write_inputs_v30(excl: set, corpus) -> dict:
    """6 dau vao chain vi30 (nguon da sach) qua bo loc an toan tap loai + ngoai kho."""
    import vi28 as v28
    import vi29 as v29
    IN30.mkdir(parents=True, exist_ok=True)
    spec = [(FILE_NAMES["pool"], OUT / "cache_pool_v30.json", "cache"),
            (FILE_NAMES["q8b"], IN29 / "q8b_v29.json", "cache"),
            (FILE_NAMES["dense"], OUT / "dense_v30_top200.json", "cache"),
            (FILE_NAMES["hyde"], OUT / "hyde_v30_top200.json", "cache"),
            (FILE_NAMES["cands"], OUT / "chunk_dw_u900_v30.json.zst", "cands"),
            (FILE_NAMES["ref_chunks"], IN29 / "ref_chunks_v29.json", "ref")]
    rep = {"tap_loai": len(excl), "files": {}}
    for name, src, kind in spec:
        obj = read(src)
        if kind == "cache":
            out = v29.filter_cache(obj, excl, corpus)
            nb, na = sum(map(len, obj.values())), sum(map(len, out.values()))
        elif kind == "cands":
            out = v29.filter_cands(obj, excl, corpus)
            nb, na = sum(map(len, obj.values())), sum(map(len, out.values()))
        else:
            out = v29.filter_ref_chunks(obj, excl, corpus)
            nb = sum(len(it.get("relevant_chunks", [])) for it in obj)
            na = sum(len(it["relevant_chunks"]) for it in out)
        dest = IN30 / name
        nbytes = v28.write_zst(dest, out) if name.endswith(".zst") else write(dest, out)
        rep["files"][name] = {"nguon": str(src), "truoc": nb, "sau": na, "bo": nb - na}
        del obj, out
    return rep


def stage_build():
    import vi29 as v29
    v29.OUT = OUT  # find_delta/chain_zip_check cua vi29 ghi vao diag30
    if (OUT / "build_report_vi30.json").exists() and (OUT / f"submission_{NAME}.zip").exists():
        print("build da xong (resume): bo qua", flush=True)
        return
    bad = load_bad()
    added = read(OUT / "new_ids.json")
    excl = build_excl(bad, added)
    corpus = v29.load_corpus_ids(LINKS)
    loc = write_inputs_v30(excl, corpus)
    print(json.dumps(loc), flush=True)
    files = v29.chain_files(IN30, FILE_NAMES)
    delta, attempts = v29.find_delta(files, "doc_text", MAX_ZIP, DELTA0, PROBE)
    r = v29.chain_zip_check(NAME, delta, MAX_ZIP, files, "doc_text", validate=True)
    assert r["code"] == 0, r
    clean = v29.assert_clean(read(OUT / f"submission_{NAME}.json"), excl, corpus)
    fp = fingerprint(OUT / f"submission_{NAME}.zip")
    write(OUT / "build_report_vi30.json",
          {"name": NAME, "delta": delta, "attempts": attempts, "max_zip": MAX_ZIP,
           "loc_dau_vao": loc, "assert_clean": clean, "validate": "ok", "dedup_mode": "doc_text",
           "zip": fp, "n_q_TB": r.get("n_q_TB"), "tong_chunk": r.get("tong_chunk")})
    print(json.dumps({"build": NAME, "delta": delta, "attempts": attempts, **clean,
                      "zip_bytes": fp["bytes"], "sha256": fp["sha256"]}), flush=True)


def stage_stats():
    import vi29 as v29
    sub = read(OUT / f"submission_{NAME}.json")
    rep30 = read(OUT / f"report_chain_{NAME}.json")
    r30 = v29.sub_row(sub, rep30, fingerprint(OUT / f"submission_{NAME}.zip"), rep30.get("delta"))
    new_ids = set(map(str, read(OUT / "new_ids.json")))
    recraw = set(map(str, read(OUT / "recraw_ids.json")))
    n = max(len(sub), 1)
    d_new = d_rc = c_new = c_rc = 0
    dl30 = {}
    for it in sub:
        docs = [str(d) for d in it["relevant_docs"]]
        dl30[str(it["id"])] = docs
        d_new += sum(d in new_ids for d in docs)
        d_rc += sum(d in recraw for d in docs)
        c_new += sum(str(c["doc_id"]) in new_ids for c in it["relevant_chunks"])
        c_rc += sum(str(c["doc_id"]) in recraw for c in it["relevant_chunks"])
    del sub
    ref = read(OUT29 / "submission_vi29.json")
    rep29 = read(OUT29 / "report_chain_vi29.json")
    r29 = v29.sub_row(ref, rep29, fingerprint(OUT29 / "submission_vi29.zip"), rep29.get("delta"))
    doi = trung30 = 0
    for it in ref:
        a = dl30.get(str(it["id"]), [])
        b = [str(d) for d in it["relevant_docs"]]
        doi += a != b
        trung30 += len(set(a[:30]) & set(b[:30]))
    del ref
    vi("newonly30", "--fhyde", OUT / "cache_fhyde_v30.json", "--new-ids", OUT / "new_ids.json",
       "--out", OUT / "submission_vi30_newonly30.json", "--report", OUT / "newonly30_report.json")
    run("evaluate.py", "validate", OUT / "submission_vi30_newonly30.json")
    run("zip_submission.py", "--json", OUT / "submission_vi30_newonly30.json",
        "--zip", OUT / "submission_vi30_newonly30.zip", "--method", "lzma")
    rep = {"vi30": r30, "vi29_tham_chieu": r29,
           "TB_doc_them_cau": round(d_new / n, 2), "TB_doc_crawl_lai_cau": round(d_rc / n, 2),
           "TB_doc_moi_han_cau": round((d_new - d_rc) / n, 2),
           "TB_chunk_tu_doc_them_cau": round(c_new / n, 2), "TB_chunk_tu_doc_crawl_lai_cau": round(c_rc / n, 2),
           "so_cau_doclist_doi": doi, "TB_top30_trung_vi29": round(trung30 / n, 2),
           "newonly30": read(OUT / "newonly30_report.json")}
    write(OUT / "stats_report_vi30.json", rep)
    write(OUT / "zip_fingerprints.json", [fingerprint(OUT / f"submission_{NAME}.zip"),
                                          fingerprint(OUT / "submission_vi30_newonly30.zip")])
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
    if lock.exists() and "vi30" in lock.read_text(errors="replace"):
        raise RuntimeError("vi30 lock still exists; inspect job before cleanup")
    name = "r2ai_vi30_" + machine
    q = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "CSV", "/NH"], capture_output=True, text=True)
    if not q.returncode:
        assert '"Running"' not in q.stdout, name + " still running"
        subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], check=True)
    print("CLEANUP DONE", machine, {p.name: p.read_text(errors="replace") for p in (ROOT / "locks").glob("*.lock")})


B660_STAGES = ["prep", "sanity", "index", "exports", "bm25", "pool", "doclist", "chunk", "build", "stats"]
Z87_STAGES = ["embed", "rr", "docwin"]


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
    with owned_lock(lock, f"vi30 {args.stage} pid={os.getpid()}"):
        globals()["stage_" + args.stage]()
    print("DONE", args.stage, flush=True)


if __name__ == "__main__":
    main()
