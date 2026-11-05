r"""vi32 = vi31 voi DUNG MOT thay doi: diem 8B q8b_v31 -> diag32\q8b_v32.json
(bu 61.862 cap thieu 8B trong top-170 fhyde_v31; scripts/q8b32.py).

Tai dung vi31.py: gan lai cac duong dan module-level (OUT, G, Q8_V31, CHUNK30/31, NAME...).
Khac vi31:
  sanity   duong vi32 voi q8b_v31 + chunk_v31 -> phai tai lap submission_vi31 1200/1200 (delta 57).
  doclist  nhu vi31 nhung (a) tap da phu docwin them doclist_v31; (b) xuat text cho MOI cap fresh
           (ke ca doc new_ids, vi doc moi lot doclist sau khi doi 8B co the khong co trong
           docs_vi30_docwin). File xuat dat ten docs_vi31_old_v32*.jsonl.zst de khop glob cua vi31.
Cac stage chunk/build/stats/docwin dung nguyen ham cua vi31.
Out b660 D:\r2ai\data\sub\vi32\, z87 D:\r2ai\work\vi32\. Artefact vi30/vi31/diag* CHI DOC.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi30
import vi31
from vi24 import read, write, run, vi, owned_lock, missing_docwin

ROOT = vi31.ROOT
OUT31 = ROOT / "data/sub/vi31"                       # CHI DOC
Q8_OLD = ROOT / "data/sub/diag31/q8b_v31.json"       # dau vao 8B cua vi31
Q8_NEW = ROOT / "data/sub/diag32/q8b_v32.json"
CHUNK_V31 = OUT31 / "chunk_dw_u900_v31.json.zst"
OUT = ROOT / "data/sub/vi32"
G = ROOT / "work/vi32"
NAME, SANITY_NAME, PROBE = "vi32", "vi32_sanity", "vi32p"
SANITY_DELTA = 57      # delta cua submission_vi31

BINDINGS = {"OUT": OUT, "IN31": OUT / "in1", "IN_S": OUT / "in_sanity", "G": G,
            "Q8_V31": Q8_NEW,
            "CHUNK30": CHUNK_V31,                  # chunk nen = chunk cua vi31
            "CHUNK31": OUT / "chunk_dw_u900_v32.json.zst",
            "NAME": NAME, "SANITY_NAME": SANITY_NAME, "PROBE": PROBE, "DELTA0": 57}


def bind() -> dict:
    """Gan lai bien module cua vi31 (moi ham vi31 doc bien toan cuc luc chay). Tra gia tri cu."""
    old = {k: getattr(vi31, k) for k in BINDINGS}
    for k, v in BINDINGS.items():
        setattr(vi31, k, v)
    return old


def export_all_fresh(fresh: dict) -> dict:
    """cache cho export_docs: moi cap fresh (khong bo new_ids), dang [[doc, idx, 0.0], ...]."""
    out = {q: [[*h, 0.0] for h in rows] for q, rows in fresh.items()}
    return {q: v for q, v in out.items() if v}


def stage_sanity():
    import vi29 as v29
    sr = OUT / "sanity_report.json"
    if sr.exists() and read(sr).get("ok"):
        print("sanity da xong (resume): bo qua", flush=True)
        return
    excl, corpus = vi31.load_excl_corpus()
    loc = vi31.write_inputs_v31(vi31.input_spec(Q8_OLD, CHUNK_V31), vi31.IN_S, excl, corpus)
    files = v29.chain_files(vi31.IN_S, vi31.FILE_NAMES)
    v29.run_chain(SANITY_NAME, SANITY_DELTA, files, "doc_text", OUT)
    cmp_rep = vi30.core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"),
                                     str(OUT31 / "submission_vi31.json"))
    cmp_rep["loc"] = loc
    write(sr, cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    v29.require_sanity(cmp_rep)
    (OUT / f"submission_{SANITY_NAME}.json").unlink()


def stage_doclist():
    import numpy as np
    import vi29 as v29
    excl, corpus = vi31.load_excl_corpus()
    raw = read(Q8_NEW)
    q8 = v29.filter_cache(raw, excl, corpus)
    n_raw, n_q8 = sum(map(len, raw.values())), sum(map(len, q8.values()))
    del raw
    write(OUT / "q8b_v31_clean.json", q8)        # ten giu nhu vi31 (stats/vi31 doc ten nay)
    del q8
    O30 = vi31.OUT30
    vi("fuse", "--pool", O30 / "cache_pool_v30.json", "--q8b", OUT / "q8b_v31_clean.json",
       "--dense", O30 / "dense_v30_top200.json", "--hyde", O30 / "hyde_v30_top200.json",
       "--out-rrf3d2", OUT / "cache_rrf3d2_v31.json", "--out-fhyde", OUT / "cache_fhyde_v31.json",
       "--report", OUT / "fuse_report.json")
    vi("doclist", "--pool", O30 / "cache_pool_v30.json", "--fhyde", OUT / "cache_fhyde_v31.json",
       "--ref-chunks", vi31.IN29 / "ref_chunks_v29.json", "--out", OUT / "doclist_v31.json",
       "--report", OUT / "doclist_report.json")
    dl32 = read(OUT / "doclist_v31.json")
    dl31 = read(OUT31 / "doclist_v31.json")
    wins30 = read(O30 / "docwin_vi30.json.zst")
    covered = vi31.union_doclists(read(vi31.OUT29 / "doclist_vi29.json"), read(O30 / "doclist_v30.json"),
                                  {q: list(v) for q, v in wins30.items()}, dl31)
    del wins30
    idx = {str(d): i for i, d in enumerate(np.load(vi31.IX / "ids.npy"))}
    chunks = read(CHUNK_V31)
    cands = missing_docwin(dl32, chunks, covered, idx)
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
            "--out-docs", OUT / "docs_vi31_old_v32.jsonl.zst", "--out-cands", OUT / "export_old_cands.json")
    n_fresh = sum(map(len, fresh.values()))
    n_new = sum(str(h[0]) in new_ids for rs in fresh.values() for h in rs)
    rep = {"q8b_v32_hit": n_raw, "q8b_v32_hit_sau_loc": n_q8,
           "doclist_vs_vi31": vi31.doclist_diff(dl32, dl31),
           "cap_chua_docwin": sum(map(len, cands.values())),
           "pairs": n_fresh, "new_doc_pairs": n_new, "old_doc_pairs": n_fresh - n_new,
           "p3_reused_pairs": sum(map(len, reuse.values())), "cau_co_cap_moi": len(fresh)}
    write(OUT / "docwin_cands_report.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


STAGES = {"sanity": stage_sanity, "doclist": stage_doclist, "chunk": vi31.stage_chunk,
          "build": vi31.stage_build, "stats": vi31.stage_stats, "docwin": vi31.stage_docwin}


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
        with owned_lock(ROOT / "locks" / "gpu.lock", f"vi32 {a.stage} pid={os.getpid()}"):
            STAGES[a.stage]()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        STAGES[a.stage]()
    print("DONE", a.stage, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
