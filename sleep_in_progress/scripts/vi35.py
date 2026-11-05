r"""vi35 = vi34 voi DUNG MOT thay doi: so doc moi cau n_q tinh theo diem hop nhat fhyde
(phuong phap "ratio": fused/fused_top1 > thr, kep [40,400]) thay cho so doc logit bge > -2,0
tren pool. thr hieu chinh MOT LAN (stage calib, CPU) sao cho TB n_q = TB n_q vi34 (166,79).

Giu y vi34: w_q8b fhyde = 2 (rrf3d2/zd chunk = 1), pool/dense/hyde v30, chi muc vi30, tham so
chain, dedup "doc_text", q8b = diag33\q8b_v33.json. Tai dung vi33 (va vi31) bang bind().
  sanity   q8b_v33 + chunk vi34, W8F 2, NQ_FUSED None -> phai tai lap submission_vi34
           1200/1200 (delta 56).
  calib    CPU: doc cache_fhyde_v31.json cua vi34 (w=2) + cache_pool_v30.json; n_cu = adaptive_n
           (-2,0/40/400); hieu chinh thr "ratio" (diag13.calibrate_fused_thr) de TB n_q moi =
           TB n_cu; ghi calib_report.json (thr, target, Spearman, phan vi...).
  doclist  nhu vi33.stage_doclist voi DOCLIST_EXTRA = --nq-fused-method ratio --nq-fused-thr thr
           (doc tu calib_report.json); sau do do phu 8B cua doclist moi + cap (cau, doc) thieu
           diem 8B -> q8b_gap_report.json {cap, thieu_8b, ty_le, cau_co_thieu} (ty_le = ty le CO
           8B; ty_le < 1.0 = co cap thieu) + q8b_gap_pairs.json {qid: [doc_id...]}.
  build    dat vi23.W_Q8B_FHYDE = 2 + vi23.NQ_FUSED = ("ratio", thr) trong luc vi31.stage_build(),
           finally tra ve None; kiem report_chain_vi35.json nq_mode=fused_ratio, n_q_TB ~ target,
           doclist_vi35.json == doclist_v31.json.
  chunk/stats/docwin  nhu vi34 (vi31).
Out b660 D:\r2ai\data\sub\vi35\, z87 D:\r2ai\work\vi35\. Artefact vi34/diag* CHI DOC.
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
import vi33
from vi24 import read, write, owned_lock

ROOT = vi31.ROOT
OUT34 = ROOT / "data/sub/vi34"                       # CHI DOC
Q8 = ROOT / "data/sub/diag33/q8b_v33.json"           # 8B nhu vi34
CHUNK_V34 = OUT34 / "chunk_dw_u900_v34.json.zst"     # chunk nen = chunk cua vi34
FHYDE34 = OUT34 / "cache_fhyde_v31.json"             # fhyde w=2 cua vi34 (calib)
POOL30 = ROOT / "data/sub/diag30/cache_pool_v30.json"
OUT = ROOT / "data/sub/vi35"
G = ROOT / "work/vi35"
NAME, SANITY_NAME, PROBE = "vi35", "vi35_sanity", "vi35p"
W8F = 2.0
DELTA = 56             # delta cua submission_vi34
NQ_METHOD = "ratio"

BINDINGS31 = {"OUT": OUT, "IN31": OUT / "in1", "IN_S": OUT / "in_sanity", "G": G,
              "Q8_V31": Q8, "CHUNK30": CHUNK_V34,
              "CHUNK31": OUT / "chunk_dw_u900_v35.json.zst",
              "NAME": NAME, "SANITY_NAME": SANITY_NAME, "PROBE": PROBE, "DELTA0": DELTA}
# vi33.stage_doclist doc cac bien nay: OUT32 -> vi34 (doclist_v31.json = doclist cu de so sanh)
BINDINGS33 = {"OUT": OUT, "G": G, "Q8": Q8, "CHUNK_V32": CHUNK_V34, "OUT32": OUT34, "W8F": W8F,
              "NAME": NAME, "SANITY_NAME": SANITY_NAME, "PROBE": PROBE, "DELTA": DELTA,
              "DOCLIST_EXTRA": []}   # gan that khi stage doclist (doc thr tu calib)


def bind() -> dict:
    """Gan lai bien module vi31 + vi33. Tra {(mod, k): gia tri cu}."""
    old = {}
    for mod, b in ((vi31, BINDINGS31), (vi33, BINDINGS33)):
        for k, v in b.items():
            old[(mod.__name__, k)] = getattr(mod, k)
            setattr(mod, k, v)
    return old


def unbind(old: dict):
    mods = {"vi31": vi31, "vi33": vi33}
    for (m, k), v in old.items():
        setattr(mods[m], k, v)


def stage_sanity():
    import vi29 as v29
    sr = OUT / "sanity_report.json"
    if sr.exists() and read(sr).get("ok"):
        print("sanity da xong (resume): bo qua", flush=True)
        return
    old_nqf = vi23.NQ_FUSED
    vi33.set_w8f(W8F)
    vi23.NQ_FUSED = None
    try:
        excl, corpus = vi31.load_excl_corpus()
        loc = vi31.write_inputs_v31(vi31.input_spec(Q8, CHUNK_V34), vi31.IN_S, excl, corpus)
        files = v29.chain_files(vi31.IN_S, vi31.FILE_NAMES)
        v29.run_chain(SANITY_NAME, DELTA, files, "doc_text", OUT)
    finally:
        vi33.set_w8f(None)
        vi23.NQ_FUSED = old_nqf
    cmp_rep = vi30.core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"),
                                     str(OUT34 / "submission_vi34.json"))
    cmp_rep["loc"] = loc
    write(sr, cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    v29.require_sanity(cmp_rep)
    (OUT / f"submission_{SANITY_NAME}.json").unlink()


def stage_calib():
    import diag13
    from union_sub import adaptive_n
    fhyde = read(FHYDE34)                 # fhyde w=2 cua vi34 = thu tu se dung trong chain
    pool = read(POOL30)
    qs = sorted(pool, key=lambda x: int(x))
    n_old = [adaptive_n(pool[q], -2.0, 40, 400) for q in qs]
    target = sum(n_old) / len(n_old)
    thr, ns = diag13.calibrate_fused_thr([fhyde.get(q, []) for q in qs], NQ_METHOD, target)
    n = max(len(ns), 1)

    def pv(vals, p):
        return diag13.pct(vals, p)

    rep = {"method": NQ_METHOD, "thr": thr, "target": target,
           "TB_n_q_moi": sum(ns) / n, "n_q_moi_min": min(ns), "n_q_moi_max": max(ns),
           "spearman_n_cu_n_moi": diag13.spearman(n_old, ns),
           "TB_chenhlech_tuyetdoi": sum(abs(a - b) for a, b in zip(n_old, ns)) / n,
           "so_cau_n_bang_40": sum(1 for x in ns if x == 40),
           "so_cau_n_bang_400": sum(1 for x in ns if x == 400),
           "phan_vi_n_cu": {"p10": pv(n_old, 0.1), "p50": pv(n_old, 0.5), "p90": pv(n_old, 0.9)},
           "phan_vi_n_moi": {"p10": pv(ns, 0.1), "p50": pv(ns, 0.5), "p90": pv(ns, 0.9)},
           "so_cau": len(ns), "TB_n_q_cu": target,
           "fhyde": str(FHYDE34), "pool": str(POOL30)}
    write(OUT / "calib_report.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def stage_doclist():
    calib = read(OUT / "calib_report.json")
    vi33.DOCLIST_EXTRA = ["--nq-fused-method", NQ_METHOD, "--nq-fused-thr", str(calib["thr"])]
    try:
        vi33.stage_doclist()
    finally:
        vi33.DOCLIST_EXTRA = []
    dl = read(OUT / "doclist_v31.json")
    q8 = read(Q8)
    cov = vi33.coverage_doclist(dl, q8)
    gaps = {}
    for q, docs in dl.items():
        have = {str(h[0]) for h in q8.get(str(q), [])}
        miss = [d for d in docs if str(d) not in have]
        if miss:
            gaps[str(q)] = miss
    thieu = sum(len(v) for v in gaps.values())
    rep = {"cap": cov["cap"], "thieu_8b": thieu,
           "ty_le": round((cov["cap"] - thieu) / max(cov["cap"], 1), 6),
           "cau_co_thieu": len(gaps)}
    write(OUT / "q8b_gap_report.json", rep)
    write(OUT / "q8b_gap_pairs.json", gaps)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def stage_build():
    calib = read(OUT / "calib_report.json")
    vi23.W_Q8B_FHYDE = W8F
    vi23.NQ_FUSED = (NQ_METHOD, calib["thr"])
    try:
        vi31.stage_build()
    finally:
        vi23.W_Q8B_FHYDE = None
        vi23.NQ_FUSED = None
    rep = read(OUT / f"report_chain_{NAME}.json")
    assert rep.get("nq_mode") == f"fused_{NQ_METHOD}", rep.get("nq_mode")
    assert abs(rep.get("n_q_TB", -1e9) - calib["target"]) < 0.5, (rep.get("n_q_TB"), calib["target"])
    dl_chain = read(OUT / f"doclist_{NAME}.json")
    dl_stage = read(OUT / "doclist_v31.json")
    norm = lambda d: {str(q): [str(x) for x in v] for q, v in d.items()}  # noqa: E731
    assert norm(dl_chain) == norm(dl_stage), "doclist chain != doclist stage"
    print(json.dumps({"kiem_tra_sau_build": "ok", "nq_mode": rep["nq_mode"],
                      "n_q_TB": rep["n_q_TB"], "target": calib["target"]}, ensure_ascii=False),
          flush=True)


STAGES = {"sanity": stage_sanity, "calib": stage_calib, "doclist": stage_doclist,
          "chunk": vi31.stage_chunk, "build": stage_build, "stats": vi31.stage_stats,
          "docwin": vi31.stage_docwin}


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
        with owned_lock(ROOT / "locks" / "gpu.lock", f"vi35 {a.stage} pid={os.getpid()}"):
            STAGES[a.stage]()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        STAGES[a.stage]()
    print("DONE", a.stage, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
