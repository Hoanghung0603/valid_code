r"""vi31 = vi30 voi DUNG MOT thay doi: diem 8B q8b_v29.json -> diag31\q8b_v31.json
(bu 38.172 cap top-100 fhyde_v30 thieu diem 8B; scripts/q8b31.py).

Moi thu khac giu y vi30: pool_v30, dense_v30, hyde_v30, chi muc vi30, ref_chunks_v29, tham so,
cong thuc chain vi29 dedup "doc_text", tap loai = bad chua crawl lai U ngoai kho.
Doi 8B lam doi fhyde -> doclist co the co cap (cau, doc) chua co diem docwin: chi cac cap
do can docwin moi (z87), roi ghep chunk vao chunk_dw_u900_v30.

Stage b660 (deploy/b660_vi31.bat, D:\r2ai\data\sub\vi31\):
  sanity   duong dau vao vi31 nhung voi q8b_v29 + chunk_v30 -> phai tai lap submission_vi30
           1200/1200 (delta 57); lech -> exit 2.
  doclist  loc q8b_v31 (tap loai + ngoai kho) -> fuse -> doclist_v31; cap chua co docwin
           (khong thuoc chunk_v30, doclist_vi29, doclist_v30, docwin_vi30) -> reuse p3 / fresh;
           xuat text doc cu cua cap fresh.
  chunk    chunk U900 cho cap moi + ghep chunk_v30 -> chunk_dw_u900_v31.
  build    find_delta tu 57, tran zip 99.000.000 B LZMA, assert_clean.
  stats    so voi vi30.
Stage z87 (deploy/z87_vi31.bat, D:\r2ai\work\vi31\): docwin (300/250, max-len 512) + wincheck.
Artefact diag29/diag30/diag31 CHI DOC.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi30
from vi24 import fingerprint, read, write, run, vi, owned_lock, gpu_run, missing_docwin, merge_chunks

ROOT = vi30.ROOT
OUT29, IN29 = vi30.OUT29, vi30.IN29                 # CHI DOC
OUT30, IN30 = vi30.OUT, vi30.IN30                   # CHI DOC
OUT31D = ROOT / "data/sub/diag31"                   # CHI DOC
Q8_V29 = IN29 / "q8b_v29.json"
Q8_V31 = OUT31D / "q8b_v31.json"
IX = vi30.IX
P3 = vi30.P3
LINKS = vi30.LINKS
OUT = ROOT / "data/sub/vi31"
IN31 = OUT / "in1"
IN_S = OUT / "in_sanity"
G = ROOT / "work/vi31"
G30 = ROOT / "work/vi30"                             # z87, CHI DOC
NAME, SANITY_NAME, PROBE = "vi31", "vi31_sanity", "vi31p"
SANITY_DELTA = 57      # delta cua submission_vi30
DELTA0 = 57
MAX_ZIP = 99_000_000
FILE_NAMES = {"pool": "pool_v31.json", "q8b": "q8b_v31.json", "dense": "dense_v31.json",
              "hyde": "hyde_v31.json", "cands": "cands_v31.json.zst",
              "ref_chunks": "ref_chunks_v31.json"}
CHUNK30 = OUT30 / "chunk_dw_u900_v30.json.zst"
CHUNK31 = OUT / "chunk_dw_u900_v31.json.zst"


# ---------------------------------------------------------------- ham thuan (co test)

def union_doclists(*dls) -> dict:
    """{q: [doc...]} hop nhieu doclist/tap da phu (giu thu tu xuat hien, khong lap)."""
    out = {}
    for dl in dls:
        for q, docs in (dl or {}).items():
            cur = out.setdefault(str(q), [])
            seen = set(map(str, cur))
            for d in docs:
                if str(d) not in seen:
                    seen.add(str(d))
                    cur.append(d)
    return out


def doclist_diff(a: dict, b: dict, top: int = 30) -> dict:
    """So doclist a (moi) voi b (cu): so cau doi, TB doc them/bot, TB trung top-k."""
    qs = sorted(set(a) | set(b), key=lambda x: int(x))
    doi = them = bot = trung = 0
    for q in qs:
        x = [str(d) for d in a.get(q, [])]
        y = [str(d) for d in b.get(q, [])]
        doi += x != y
        them += len(set(x) - set(y))
        bot += len(set(y) - set(x))
        trung += len(set(x[:top]) & set(y[:top]))
    n = max(len(qs), 1)
    return {"so_cau": len(qs), "so_cau_doi": doi, "TB_doc_them_cau": round(them / n, 3),
            "TB_doc_bot_cau": round(bot / n, 3), f"TB_top{top}_trung": round(trung / n, 3)}


def input_spec(q8b_src, cands_src) -> list:
    """6 dau vao chain: nhu vi30.write_inputs_v30, chi q8b va cands la tham so."""
    return [(FILE_NAMES["pool"], OUT30 / "cache_pool_v30.json", "cache"),
            (FILE_NAMES["q8b"], Path(q8b_src), "cache"),
            (FILE_NAMES["dense"], OUT30 / "dense_v30_top200.json", "cache"),
            (FILE_NAMES["hyde"], OUT30 / "hyde_v30_top200.json", "cache"),
            (FILE_NAMES["cands"], Path(cands_src), "cands"),
            (FILE_NAMES["ref_chunks"], IN29 / "ref_chunks_v29.json", "ref")]


def coverage_8b(fhyde: dict, q8b: dict, top: int = 100) -> dict:
    """Ty le doc top-k fhyde co diem 8B."""
    tot = co = 0
    for q, hits in fhyde.items():
        have = {str(h[0]) for h in q8b.get(str(q), [])}
        ids = [str(h[0]) for h in hits[:top]]
        tot += len(ids)
        co += sum(d in have for d in ids)
    return {"top": top, "cap": tot, "co_8b": co, "ty_le": round(co / max(tot, 1), 4)}


# ---------------------------------------------------------------- tien ich

def load_excl_corpus():
    import vi29 as v29
    excl = vi30.build_excl(vi30.load_bad(), read(OUT30 / "new_ids.json"))
    return excl, v29.load_corpus_ids(LINKS)


def write_inputs_v31(spec: list, indir: Path, excl: set, corpus) -> dict:
    import vi28 as v28
    import vi29 as v29
    indir.mkdir(parents=True, exist_ok=True)
    rep = {"tap_loai": len(excl), "out_dir": str(indir), "files": {}}
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
        dest = indir / name
        v28.write_zst(dest, out) if name.endswith(".zst") else write(dest, out)
        rep["files"][name] = {"nguon": str(src), "truoc": nb, "sau": na, "bo": nb - na}
        del obj, out
    return rep


def cands_source() -> Path:
    rep = read(OUT / "chunk_report.json")
    return CHUNK31 if rep.get("new_chunks", 0) or rep.get("ghep") else CHUNK30


# ---------------------------------------------------------------- b660

def stage_sanity():
    import vi29 as v29
    sr = OUT / "sanity_report.json"
    if sr.exists() and read(sr).get("ok"):
        print("sanity da xong (resume): bo qua", flush=True)
        return
    excl, corpus = load_excl_corpus()
    loc = write_inputs_v31(input_spec(Q8_V29, CHUNK30), IN_S, excl, corpus)
    files = v29.chain_files(IN_S, FILE_NAMES)
    v29.run_chain(SANITY_NAME, SANITY_DELTA, files, "doc_text", OUT)
    cmp_rep = vi30.core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"),
                                     str(OUT30 / "submission_vi30.json"))
    names30 = vi30.FILE_NAMES
    cmp_rep["sha_dau_vao_khop_in1_vi30"] = {
        k: fingerprint(IN_S / FILE_NAMES[k])["sha256"] == fingerprint(IN30 / names30[k])["sha256"]
        for k in FILE_NAMES}
    cmp_rep["loc"] = loc
    write(sr, cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    v29.require_sanity(cmp_rep)
    (OUT / f"submission_{SANITY_NAME}.json").unlink()


def stage_doclist():
    import numpy as np
    import vi29 as v29
    excl, corpus = load_excl_corpus()
    raw = read(Q8_V31)
    q8 = v29.filter_cache(raw, excl, corpus)
    n_raw, n_q8 = sum(map(len, raw.values())), sum(map(len, q8.values()))
    del raw
    write(OUT / "q8b_v31_clean.json", q8)
    del q8
    vi("fuse", "--pool", OUT30 / "cache_pool_v30.json", "--q8b", OUT / "q8b_v31_clean.json",
       "--dense", OUT30 / "dense_v30_top200.json", "--hyde", OUT30 / "hyde_v30_top200.json",
       "--out-rrf3d2", OUT / "cache_rrf3d2_v31.json", "--out-fhyde", OUT / "cache_fhyde_v31.json",
       "--report", OUT / "fuse_report.json")
    vi("doclist", "--pool", OUT30 / "cache_pool_v30.json", "--fhyde", OUT / "cache_fhyde_v31.json",
       "--ref-chunks", IN29 / "ref_chunks_v29.json", "--out", OUT / "doclist_v31.json",
       "--report", OUT / "doclist_report.json")
    dl31 = read(OUT / "doclist_v31.json")
    dl30 = read(OUT30 / "doclist_v30.json")
    wins30 = read(OUT30 / "docwin_vi30.json.zst")
    covered = union_doclists(read(OUT29 / "doclist_vi29.json"), dl30,
                             {q: list(v) for q, v in wins30.items()})
    del wins30
    idx = {str(d): i for i, d in enumerate(np.load(IX / "ids.npy"))}
    chunks30 = read(CHUNK30)
    cands = missing_docwin(dl31, chunks30, covered, idx)
    del chunks30
    new_ids = set(map(str, read(OUT30 / "new_ids.json")))
    no_reuse = new_ids | {str(i) for i in vi30.load_bad()}
    reuse, fresh = vi30.split_docwin_cands(cands, read(P3 / "docwin_p3.json.zst"), no_reuse)
    write(OUT / "cands_docwin_vi31.json", fresh)
    write(OUT / "cands_docwin_reuse.json", reuse)
    old_rows = {q: [[*h, 0.0] for h in rows if str(h[0]) not in new_ids] for q, rows in fresh.items()}
    old_rows = {q: v for q, v in old_rows.items() if v}
    write(OUT / "cache_docwin_old.json", old_rows)
    if old_rows:
        run("export_docs.py", "--index", IX, "--cache", OUT / "cache_docwin_old.json",
            "--n", 1000, "--chars", 200000, "--no-title", "--part-mb", 80,
            "--out-docs", OUT / "docs_vi31_old.jsonl.zst", "--out-cands", OUT / "export_old_cands.json")
    n_fresh = sum(map(len, fresh.values()))
    n_new = sum(str(h[0]) in new_ids for rows in fresh.values() for h in rows)
    rep = {"q8b_v31_hit": n_raw, "q8b_v31_hit_sau_loc": n_q8,
           "doclist_vs_vi30": doclist_diff(dl31, dl30),
           "cap_chua_docwin": sum(map(len, cands.values())),
           "pairs": n_fresh, "new_doc_pairs": n_new, "old_doc_pairs": n_fresh - n_new,
           "p3_reused_pairs": sum(map(len, reuse.values())), "cau_co_cap_moi": len(fresh)}
    write(OUT / "docwin_cands_report.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def stage_chunk():
    from chunks_from_docwin import build_q_rows, select_q_rows, verify_verbatim
    from dwx_merge import write_zst_stream
    from rerank_run import load_docs
    cands, reuse = read(OUT / "cands_docwin_vi31.json"), read(OUT / "cands_docwin_reuse.json")
    if not cands and not reuse:
        rep = {"new_chunks": 0, "ghep": False, "ghi_chu": "khong co cap moi: dung chunk_dw_u900_v30"}
        write(OUT / "chunk_report.json", rep)
        print(json.dumps(rep), flush=True)
        return
    wins = read(OUT / "docwin_vi31.json.zst") if cands else {}
    p3 = read(P3 / "docwin_p3.json.zst")
    for q, rows in reuse.items():
        cands.setdefault(q, []).extend(rows)
        wins.setdefault(q, {}).update({str(h[0]): p3[q][str(h[0])] for h in rows})
    del p3
    paths = [OUT30 / "docs_vi30_docwin.jsonl.zst", *sorted(OUT.glob("docs_vi31_old*.jsonl.zst")),
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
    old = read(CHUNK30)
    merged = merge_chunks(old, new)
    write_zst_stream(CHUNK31, merged)
    rep = {"new_chunks": sum(map(len, new.values())), "ghep": True, "verbatim_bad": nbad,
           "empty_text_pairs": sum(not docs[h[0]].strip() for rows in cands.values() for h in rows),
           "old_chunks": sum(map(len, old.values())), "total_chunks": sum(map(len, merged.values()))}
    write(OUT / "chunk_report.json", rep)
    print(json.dumps(rep), flush=True)


def stage_build():
    import vi29 as v29
    v29.OUT = OUT  # find_delta/chain_zip_check cua vi29 ghi vao vi31
    if (OUT / "build_report_vi31.json").exists() and (OUT / f"submission_{NAME}.zip").exists():
        print("build da xong (resume): bo qua", flush=True)
        return
    excl, corpus = load_excl_corpus()
    src = cands_source()
    loc = write_inputs_v31(input_spec(Q8_V31, src), IN31, excl, corpus)
    print(json.dumps(loc), flush=True)
    files = v29.chain_files(IN31, FILE_NAMES)
    delta, attempts = v29.find_delta(files, "doc_text", MAX_ZIP, DELTA0, PROBE)
    r = v29.chain_zip_check(NAME, delta, MAX_ZIP, files, "doc_text", validate=True)
    assert r["code"] == 0, r
    clean = v29.assert_clean(read(OUT / f"submission_{NAME}.json"), excl, corpus)
    fp = fingerprint(OUT / f"submission_{NAME}.zip")
    write(OUT / "build_report_vi31.json",
          {"name": NAME, "delta": delta, "attempts": attempts, "max_zip": MAX_ZIP,
           "q8b": str(Q8_V31), "cands": str(src), "loc_dau_vao": loc, "assert_clean": clean,
           "validate": "ok", "dedup_mode": "doc_text", "zip": fp,
           "n_q_TB": r.get("n_q_TB"), "tong_chunk": r.get("tong_chunk")})
    print(json.dumps({"build": NAME, "delta": delta, "attempts": attempts, **clean,
                      "zip_bytes": fp["bytes"], "sha256": fp["sha256"]}), flush=True)


def _digest(sub: list) -> tuple:
    dl = {str(it["id"]): [str(d) for d in it["relevant_docs"]] for it in sub}
    ch = {str(it["id"]): {(str(c["doc_id"]), c["chunk_text"]) for c in it["relevant_chunks"]} for it in sub}
    return dl, ch


def stage_stats():
    import vi29 as v29
    sub = read(OUT / f"submission_{NAME}.json")
    rep31 = read(OUT / f"report_chain_{NAME}.json")
    r31 = v29.sub_row(sub, rep31, fingerprint(OUT / f"submission_{NAME}.zip"), rep31.get("delta"))
    dl31, ch31 = _digest(sub)
    del sub
    sub = read(OUT30 / "submission_vi30.json")
    rep30 = read(OUT30 / "report_chain_vi30.json")
    r30 = v29.sub_row(sub, rep30, fingerprint(OUT30 / "submission_vi30.zip"), rep30.get("delta"))
    dl30, ch30 = _digest(sub)
    del sub
    n = max(len(ch31), 1)
    moi = sum(len(ch31[q] - ch30.get(q, set())) for q in ch31)
    chung = sum(len(ch31[q] & ch30.get(q, set())) for q in ch31)
    fh = read(OUT / "cache_fhyde_v31.json")
    rep = {"vi31": r31, "vi30_tham_chieu": r30,
           "doclist_vs_vi30": doclist_diff(dl31, dl30),
           "doclist_chain_khop_doclist_v31": dl31 == {q: [str(d) for d in v]
                                                    for q, v in read(OUT / "doclist_v31.json").items()},
           "TB_chunk_moi_cau": round(moi / n, 2), "TB_chunk_chung_cau": round(chung / n, 2),
           "so_cau_chunk_doi": sum(ch31[q] != ch30.get(q, set()) for q in ch31),
           "phu_8b_top100_v31": coverage_8b(fh, read(OUT / "q8b_v31_clean.json")),
           "phu_8b_top100_v30": coverage_8b(read(OUT30 / "cache_fhyde_v30.json"), read(IN30 / "q8b_v30.json"))}
    write(OUT / "stats_report_vi31.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- z87

def stage_docwin():
    if not (G / "wincheck.json").exists():
        gpu_run("dwx_p3.py", "winsample", "--docwin", ROOT / "work/docwin_ext/docwin_ext.json.zst",
                "--docs", ROOT / "work/docwin_ext/docwin_ext.part000.jsonl.zst",
                ROOT / "work/docwin_ext/docwin_ext.part001.jsonl.zst", "--queries", ROOT / "work/queries.jsonl",
                "--out", G / "winsample.json", "--n", vi30.WIN_N)
        gpu_run("dwx_p3.py", "score", "--in", G / "winsample.json", "--out", G / "winscore.json",
                "--model", "BAAI/bge-reranker-v2-m3", "--max-len", 512, "--batch", 16)
        gpu_run("dwx_p3.py", "wincheck", "--ref", G / "winsample.json", "--got", G / "winscore.json",
                "--report", G / "wincheck.json", "--tol", vi30.WIN_TOL)
    vi30.check_wincheck(read(G / "wincheck.json"))
    docs = [G30 / "docs_vi30_docwin.jsonl.zst"] + sorted(G.glob("docs_vi31_old*.jsonl.zst"))
    gpu_run("docwin_run.py", "--docs", *docs, "--queries", ROOT / "work/queries.jsonl",
            "--cands", G / "cands_docwin_vi31.json", "--out", G / "docwin_vi31.json.zst",
            "--max-len", 512, "--batch", 16, "--top-docs", 500, "--window-words", 300,
            "--window-stride", 250, "--resume-dir", G / "dw_parts")


# ---------------------------------------------------------------- main

B660_STAGES = ["sanity", "doclist", "chunk", "build", "stats"]
Z87_STAGES = ["docwin"]


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=B660_STAGES + Z87_STAGES)
    parser.add_argument("--machine", choices=["b660", "z87"], default="b660")
    args = parser.parse_args()
    t0 = time.time()
    if args.machine == "z87":
        G.mkdir(parents=True, exist_ok=True)
        with owned_lock(ROOT / "locks" / "gpu.lock", f"vi31 {args.stage} pid={os.getpid()}"):
            globals()["stage_" + args.stage]()
    else:
        OUT.mkdir(parents=True, exist_ok=True)   # heavy_job.lock do b660_vi31.bat quan ly
        globals()["stage_" + args.stage]()
    print("DONE", args.stage, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
