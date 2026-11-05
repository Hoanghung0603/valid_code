r"""Chan doan pool runner vs pool lich su (vi30) tren tap cau smoke (CPU, khong GPU).

Gia thiet can kiem: pool runner nho hon pool lich su -> it doc co logit bge > -2 -> doclist ngan hon vi34.

3a. Moi cau: |pool runner|, |pool v30|, |giao|, n_q runner, n_q v30 (logit bge > -2, kep [40,400],
    sau loc bad + ngoai kho; cung quy tac rc.n_q_for).
3b. Doc chi co trong doclist vi34 (doclist_vi34.json), phan loai:
    (iii) khong co trong pool v30  -> phai den tu ref_chunks cf4 (doclist vi34 = fhyde(pool v30) + ref;
          fhyde chi xep hang doc trong pool, nen doc ngoai pool v30 chi co the den tu ref).
    (ii)  co trong pool v30, khong co trong pool runner -> ghi hang trong BM25-300 / dense-200 / HyDE-200 runner.
    (i)   co trong ca hai pool nhung khong vao doclist runner -> logit runner <= -2 hoay khong lot top n_q.
3c. Bien the pool runner (chi dem, logit proxy = logit v30):
    P1 = BM25-300 U dense-200 U HyDE-200 ; P2 = P1 U dense-300 (neu co file, khong thi bo).
    Moi bien the: |pool|, n_q, |doclist ~ ∩ vi34|, so doc can rerank them/cau (ngoai suy 1.200 cau, 233 cap/s).

Chay tren b660 (thu muc runner\scripts co runner_core, runner_private, evaluate):
  python runner_diag_pool.py --tag rsmoke20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner_core as rc  # noqa: E402

ROOT = Path("D:/r2ai")
RUN = ROOT / "data/sub/runner"
POOL_V30 = ROOT / "data/sub/diag30/cache_pool_v30.json"
VI34_DL = ROOT / "data/sub/vi34/doclist_vi34.json"
RATE_CAP = 233.0  # cap/s GPU bge rerank


def load(p):
    return rc.load_json(p)


def ranks(hits):
    return {str(h[0]): i + 1 for i, h in enumerate(hits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rsmoke20")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = RUN / a.tag
    import runner_private as rp  # bad + corpus cung nguon voi build
    bad = {str(x) for x in rp.load_bad()}
    corpus = {str(x) for x in rp.load_corpus()}
    ok = lambda doc: str(doc) not in bad and str(doc) in corpus  # noqa: E731

    pool_r = load(d / "pool.json")                  # runner: BM25-200 U dense-200, logit bge
    pool_h = load(POOL_V30)                         # lich su vi30 (logit bge)
    bm300 = load(d / "cache_k300.json")             # runner BM25 top-300
    dn200 = load(d / "dense_top200.json")
    hy200 = load(d / "hyde_top200.json")
    fh_r = load(d / "cache_fhyde.json")
    dl_r = load(d / "doclist.json")
    dl34 = load(VI34_DL)
    dn300_p = d / "dense_top300.json"
    dn300 = load(dn300_p) if dn300_p.exists() else None

    qs = sorted(set(pool_r) & set(dl34), key=lambda x: int(x))
    per_q, cat = {}, {"iii_ref": 0, "ii_khong_trong_pool_runner": 0, "i_trong_pool_nhung_loai": 0}
    rank_ii = {"bm25_300": 0, "dense_200": 0, "hyde_200": 0, "khong_o_dau": 0}
    tot_missing = 0
    rows = []
    for q in qs:
        hr = [h for h in pool_r[q] if ok(h[0])]
        hh = [h for h in pool_h.get(q, []) if ok(h[0])]
        sr = {str(h[0]) for h in hr}
        sh = {str(h[0]) for h in hh}
        n_r = rc.n_q_for(hr)
        n_h = rc.n_q_for(hh)
        dl = [str(x) for x in dl34.get(q, [])]
        dr = {str(x) for x in dl_r.get(q, [])}
        only34 = [x for x in dl if x not in dr]
        tot_missing += len(only34)
        for x in only34:
            if x not in sh:
                cat["iii_ref"] += 1
            elif x not in sr:
                cat["ii_khong_trong_pool_runner"] += 1
                r_bm = ranks(bm300.get(q, []))
                r_dn = ranks(dn200.get(q, []))
                r_hy = ranks(hy200.get(q, []))
                if x in r_bm:
                    rank_ii["bm25_300"] += 1
                elif x in r_dn:
                    rank_ii["dense_200"] += 1
                elif x in r_hy:
                    rank_ii["hyde_200"] += 1
                else:
                    rank_ii["khong_o_dau"] += 1
            else:
                cat["i_trong_pool_nhung_loai"] += 1
        # bien the pool
        p1 = {}
        for src in (bm300.get(q, []), dn200.get(q, []), hy200.get(q, [])):
            for h in src:
                if ok(h[0]):
                    p1.setdefault(str(h[0]), h[1])
        p2 = dict(p1)
        if dn300 is not None:
            for h in dn300.get(q, []):
                if ok(h[0]):
                    p2.setdefault(str(h[0]), h[1])
        lg_h = {str(h[0]): h[2] for h in hh}
        lg_r = {str(h[0]): h[2] for h in hr}

        def variant(pset):
            # logit: runner neu co, neu khong thi proxy v30; doc khong co logit ca hai -> khong dem vao n_q
            vals = []
            for doc in pset:
                lg = lg_r.get(doc, lg_h.get(doc))
                if lg is not None:
                    vals.append((doc, lg))
            n = min(max(sum(1 for _, lg in vals if lg > -2.0), 40), 400)
            return {"pool": len(pset), "n_q": n,
                    "can_rerank_them": len(set(pset) - sr),
                    "doclist_cap_vi34": len(set(dl) & {doc for doc, _ in sorted(vals, key=lambda t: -t[1])[:n]})}

        rows.append({"id": int(q), "pool_runner": len(sr), "pool_v30": len(sh),
                     "giao": len(sr & sh), "n_q_runner": rc.n_q_for(hr), "n_q_v30": n_h,
                     "doclist_runner": len(dr), "doclist_vi34": len(dl),
                     "doclist_giao": len(dr & set(dl)),
                     "top30_trung": len(set(map(str, dl_r.get(q, [])[:30])) & set(dl[:30])),
                     "bien_P1": variant(p1), "bien_P2": variant(p2) if dn300 is not None else None})

    def tb(key, sub=None):
        xs = [(r[key] if sub is None else r[sub][key]) for r in rows if (sub is None or r[sub] is not None)]
        return round(sum(xs) / max(len(xs), 1), 2)

    n = max(len(rows), 1)
    summary = {
        "3a": {k: tb(k) for k in ["pool_runner", "pool_v30", "giao", "n_q_runner", "n_q_v30",
                                  "doclist_runner", "doclist_vi34", "doclist_giao", "top30_trung"]},
        "3b_TB_moi_cau": {k: round(v / n, 2) for k, v in cat.items()},
        "3b_hang_ii": {k: round(v / n, 2) for k, v in rank_ii.items()},
        "3b_tong_doc_chi_vi34": tot_missing,
        "3c": {},
        "so_cau": len(rows),
        "dense300": "co" if dn300 is not None else "khong co (P2 bo)",
        "gia_tri_proxy": "logit v30 khi thieu logit runner; doc khong co ca hai khong vao n_q",
    }
    for name in ("bien_P1", "bien_P2"):
        if name == "bien_P2" and dn300 is None:
            continue
        summary["3c"][name] = {
            "pool_TB": round(sum(r[name]["pool"] for r in rows) / n, 2),
            "n_q_TB": round(sum(r[name]["n_q"] for r in rows) / n, 2),
            "doclist_cap_vi34_TB": round(sum(r[name]["doclist_cap_vi34"] for r in rows) / n, 2),
            "can_rerank_them_TB": round(sum(r[name]["can_rerank_them"] for r in rows) / n, 2),
            "cap_ngoai_suy_1200": round(sum(r[name]["can_rerank_them"] for r in rows) / n * 1200),
            "gpu_s_ngoai_suy_1200": round(sum(r[name]["can_rerank_them"] for r in rows) / n * 1200 / RATE_CAP, 1),
        }
    out = Path(a.out) if a.out else d / "diag_pool_rsmoke20.json"
    out.write_text(json.dumps({"tom_tat": summary, "cau": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
