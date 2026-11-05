r"""vi33 = vi32 voi DUNG MOT thay doi: trong so 8B trong fhyde (thu tu chon doc) 1 -> 2.

rrf3d2 (nguon zd khi chon chunk) giu W_Q8B = 1, nhu thiet ke split. Ly do: 1:1:2:1 chon 10/10 khi
8B chi phu ~2/3 top-100; vi32 phu 8B ~99% top-170, w_q8b chua tung duoc thu lai.

Tai dung vi31.py qua bind() (nhu vi32). Khac vi32:
  sanity   duong vi33 voi W_Q8B_FHYDE = None (= 1) -> phai tai lap submission_vi32 1200/1200 (delta 56).
  doclist  fuse --w-q8b-fhyde 2; cap chua docwin = doclist moi tru (chunk_v32 U doclist vi29/v30/
           docwin_vi30/v31/v32); bao cao them doclist_vs_vi32 va phu 8B cua doclist moi.
  build    dat vi23.W_Q8B_FHYDE = 2 trong luc chay chain (vi29.chain_build_v29 doc bien nay).
Out b660 D:\r2ai\data\sub\vi33\, z87 D:\r2ai\work\vi33\. Artefact vi30/vi31/vi32/diag* CHI DOC.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi23
import vi30
import vi31
from vi24 import read, write, run, vi, owned_lock, missing_docwin

ROOT = vi31.ROOT
OUT31 = ROOT / "data/sub/vi31"                       # CHI DOC
OUT32 = ROOT / "data/sub/vi32"                       # CHI DOC
Q8 = ROOT / "data/sub/diag32/q8b_v32.json"           # 8B nhu vi32
CHUNK_V32 = OUT32 / "chunk_dw_u900_v32.json.zst"
OUT = ROOT / "data/sub/vi33"
G = ROOT / "work/vi33"
NAME, SANITY_NAME, PROBE = "vi33", "vi33_sanity", "vi33p"
W8F = 2.0              # trong so 8B moi trong fhyde
DELTA = 56             # delta cua submission_vi32
DOCLIST_EXTRA = []     # vi35: them co chon n_q fused cho vi("doclist", ...); mac dinh rong

BINDINGS = {"OUT": OUT, "IN31": OUT / "in1", "IN_S": OUT / "in_sanity", "G": G,
            "Q8_V31": Q8,
            "CHUNK30": CHUNK_V32,                  # chunk nen = chunk cua vi32
            "CHUNK31": OUT / "chunk_dw_u900_v33.json.zst",
            "NAME": NAME, "SANITY_NAME": SANITY_NAME, "PROBE": PROBE, "DELTA0": DELTA}


def bind() -> dict:
    """Gan lai bien module cua vi31. Tra gia tri cu."""
    old = {k: getattr(vi31, k) for k in BINDINGS}
    for k, v in BINDINGS.items():
        setattr(vi31, k, v)
    return old


def set_w8f(w):
    """Dat trong so 8B rieng cho fhyde trong chain (None = nhu cu)."""
    vi23.W_Q8B_FHYDE = w


def coverage_doclist(doclist: dict, q8b: dict) -> dict:
    """Ty le doc trong doclist co diem 8B."""
    tot = co = 0
    for q, docs in doclist.items():
        have = {str(h[0]) for h in q8b.get(str(q), [])}
        tot += len(docs)
        co += sum(str(d) in have for d in docs)
    return {"cap": tot, "co_8b": co, "ty_le": round(co / max(tot, 1), 4)}


def export_all_fresh(fresh: dict) -> dict:
    out = {q: [[*h, 0.0] for h in rows] for q, rows in fresh.items()}
    return {q: v for q, v in out.items() if v}


def stage_sanity():
    import vi29 as v29
    sr = OUT / "sanity_report.json"
    if sr.exists() and read(sr).get("ok"):
        print("sanity da xong (resume): bo qua", flush=True)
        return
    set_w8f(None)
    excl, corpus = vi31.load_excl_corpus()
    loc = vi31.write_inputs_v31(vi31.input_spec(Q8, CHUNK_V32), vi31.IN_S, excl, corpus)
    files = v29.chain_files(vi31.IN_S, vi31.FILE_NAMES)
    v29.run_chain(SANITY_NAME, DELTA, files, "doc_text", OUT)
    cmp_rep = vi30.core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"),
                                     str(OUT32 / "submission_vi32.json"))
    cmp_rep["loc"] = loc
    write(sr, cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    v29.require_sanity(cmp_rep)
    (OUT / f"submission_{SANITY_NAME}.json").unlink()


def stage_doclist():
    import numpy as np
    import vi29 as v29
    excl, corpus = vi31.load_excl_corpus()
    q8 = v29.filter_cache(read(Q8), excl, corpus)
    write(OUT / "q8b_v31_clean.json", q8)        # ten giu nhu vi31 (stats doc ten nay)
    O30 = vi31.OUT30
    vi("fuse", "--pool", O30 / "cache_pool_v30.json", "--q8b", OUT / "q8b_v31_clean.json",
       "--dense", O30 / "dense_v30_top200.json", "--hyde", O30 / "hyde_v30_top200.json",
       "--out-rrf3d2", OUT / "cache_rrf3d2_v31.json", "--out-fhyde", OUT / "cache_fhyde_v31.json",
       "--w-q8b-fhyde", W8F, "--report", OUT / "fuse_report.json")
    vi("doclist", "--pool", O30 / "cache_pool_v30.json", "--fhyde", OUT / "cache_fhyde_v31.json",
       "--ref-chunks", vi31.IN29 / "ref_chunks_v29.json", "--out", OUT / "doclist_v31.json",
       "--report", OUT / "doclist_report.json", *DOCLIST_EXTRA)
    dl33 = read(OUT / "doclist_v31.json")
    dl32 = read(OUT32 / "doclist_v31.json")
    wins30 = read(O30 / "docwin_vi30.json.zst")
    covered = vi31.union_doclists(read(vi31.OUT29 / "doclist_vi29.json"), read(O30 / "doclist_v30.json"),
                                  {q: list(v) for q, v in wins30.items()},
                                  read(OUT31 / "doclist_v31.json"), dl32)
    del wins30
    idx = {str(d): i for i, d in enumerate(np.load(vi31.IX / "ids.npy"))}
    chunks = read(CHUNK_V32)
    cands = missing_docwin(dl33, chunks, covered, idx)
    del chunks
    new_ids = set(map(str, read(O30 / "new_ids.json")))
    no_reuse = new_ids | {str(i) for i in vi30.load_bad()}
    reuse, fresh = vi30.split_docwin_cands(cands, read(vi31.P3 / "docwin_p3.json.zst"), no_reuse)
    write(OUT / "cands_docwin_vi31.json", fresh)
    write(OUT / "cands_docwin_reuse.json", reuse)
    rows = export_all_fresh(fresh)
    write(OUT / "cache_docwin_old.json", rows)
    if rows:
        run("export_docs.py", "--index", vi31.IX, "--cache", OUT / "cache_docwin_old.json",
            "--n", 1000, "--chars", 200000, "--no-title", "--part-mb", 80,
            "--out-docs", OUT / "docs_vi31_old_v33.jsonl.zst", "--out-cands", OUT / "export_old_cands.json")
    n_fresh = sum(map(len, fresh.values()))
    rep = {"w_q8b_fhyde": W8F,
           "doclist_vs_vi32": vi31.doclist_diff(dl33, dl32),
           "doclist_vs_vi32_top100": vi31.doclist_diff(dl33, dl32, 100),
           "phu_8b_doclist_v33": coverage_doclist(dl33, q8),
           "phu_8b_doclist_v32": coverage_doclist(dl32, q8),
           "cap_chua_docwin": sum(map(len, cands.values())),
           "pairs": n_fresh, "p3_reused_pairs": sum(map(len, reuse.values())),
           "cau_co_cap_moi": len(fresh)}
    write(OUT / "docwin_cands_report.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def stage_build():
    set_w8f(W8F)
    try:
        vi31.stage_build()
    finally:
        set_w8f(None)


STAGES = {"sanity": stage_sanity, "doclist": stage_doclist, "chunk": vi31.stage_chunk,
          "build": stage_build, "stats": vi31.stage_stats, "docwin": vi31.stage_docwin}


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--machine", choices=["b660", "z87"], default="b660")
    a = ap.parse_args()
    bind()
    t0 = time.time()
    if a.machine == "z87":
        G.mkdir(parents=True, exist_ok=True)
        with owned_lock(ROOT / "locks" / "gpu.lock", f"vi33 {a.stage} pid={os.getpid()}"):
            STAGES[a.stage]()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        STAGES[a.stage]()
    print("DONE", a.stage, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
