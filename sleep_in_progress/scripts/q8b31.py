r"""q8b31: cham Qwen3-Reranker-8B cho cap (cau, doc) trong top-100 cache_fhyde_v30 (chi doc thuoc
pool loc in1/pool_v30) ma q8b_v29 CHUA co diem, roi gop thanh q8b_v31.

select  (b660) chon cap + xuat text giong lan 10/10 (chunkdocs100 = export_docs --no-title
        --chars 200000 tren chi muc; o day chi muc vi30, cat them con 12.288 ky tu = max_len*12,
        dung bang muc cat cua qwen_rerank_run.score_pairs nen diem khong doi) + queries.jsonl
        + cands shard theo dinh dang qwen_rerank_run ({qid: [[doc_id, idx], ...]}) + bench
        (cap DA co diem cu, de kiem cung thang).
merge   (b660/local) q8b_v31 = moi cau hop hit cu q8b_v29 + hit moi, sap giam dan; kiem tra.

Dung:
  python scripts/q8b31.py select --outdir D:\r2ai\data\sub\diag31\q8b31_in --shards 2 --bench 200
  python scripts/q8b31.py merge --old q8b_v29.json --new s0.json s1.json --pool pool_v30.json
      --new-ids new_ids.json --out q8b_v31.json --report q8b31_merge.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
OUT30 = ROOT / "data/sub/diag30"
IN29 = ROOT / "data/sub/diag29/in1"
IX = ROOT / "data/index/vi30"
OUT31 = ROOT / "data/sub/diag31"
TOP = 100
SELECT_FHYDE = OUT30 / "cache_fhyde_v30.json"
SELECT_POOL = OUT30 / "in1/pool_v30.json"
SELECT_Q8B = IN29 / "q8b_v29.json"
SELECT_NEW_IDS = OUT30 / "new_ids.json"
PREFIX = "q8b31"
CHARS = 200000      # nhu chunkdocs100 (10/10)
CAP = 1024 * 12     # qwen_rerank_run: max_len * DOC_CHARS_PER_TOKEN


# ---------------------------------------------------------------- ham thuan (co test)

def select_pairs(fhyde: dict, pool: dict, q8b: dict, top: int = TOP) -> dict:
    """{qid: [doc_id...]} = top-`top` fhyde, chi doc thuoc pool[q], bo doc da co diem q8b[q],
    giu thu tu fhyde, khong trung."""
    po = {str(q): {str(h[0]) for h in v} for q, v in pool.items()}
    old = {str(q): {str(h[0]) for h in v} for q, v in q8b.items()}
    out = {}
    for q, hits in fhyde.items():
        q = str(q)
        ps, os_, seen, keep = po.get(q, set()), old.get(q, set()), set(), []
        for h in hits[:top]:
            d = str(h[0])
            if d in seen or d not in ps or d in os_:
                continue
            seen.add(d)
            keep.append(int(h[0]))
        out[q] = keep
    return out


def bench_sample(fhyde: dict, q8b: dict, n: int, seed: int = 31, top: int = TOP) -> list:
    """n cap (qid, doc_id, diem_cu) ngau nhien: doc trong top-`top` fhyde DA co diem q8b."""
    cand = []
    for q, hits in fhyde.items():
        sc = {str(h[0]): h[2] for h in q8b.get(str(q), []) if h[2] is not None}
        for h in hits[:top]:
            if str(h[0]) in sc:
                cand.append((str(q), int(h[0]), sc[str(h[0])]))
    cand = sorted(set(cand))
    random.Random(seed).shuffle(cand)
    return cand[:n]


def split_shards(pairs: dict, k: int) -> list:
    """Chia cau thanh k shard can bang so cap (tham lam, cau nhieu cap truoc)."""
    shards = [dict() for _ in range(k)]
    load = [0] * k
    for q in sorted(pairs, key=lambda x: (-len(pairs[x]), int(x))):
        if not pairs[q]:
            continue
        i = load.index(min(load))
        shards[i][q] = pairs[q]
        load[i] += len(pairs[q])
    return [dict(sorted(s.items(), key=lambda kv: int(kv[0]))) for s in shards]


def merge_q8b(old: dict, new: dict) -> tuple:
    """old/new {qid: [[doc, idx, score], ...]}; hit moi score None bi bo. Moi cau: hop, doc trung
    giu diem CU, sap giam dan (on dinh). Tra (out, {"trung_bo": n, "none_bo": n})."""
    old = {str(q): v for q, v in old.items()}
    new = {str(q): v for q, v in new.items()}
    out, dup, none = {}, 0, 0
    for q in sorted(set(old) | set(new), key=int):
        rows = [list(h) for h in old.get(q, [])]
        seen = {str(h[0]) for h in rows}
        for h in new.get(q, []):
            if h[2] is None:
                none += 1
                continue
            if str(h[0]) in seen:
                dup += 1
                continue
            seen.add(str(h[0]))
            rows.append(list(h))
        rows.sort(key=lambda h: -h[2])
        out[q] = rows
    return out, {"trung_bo": dup, "none_bo": none}


def check_merged(out: dict, new: dict, pool: dict, n_q: int = 1200) -> dict:
    po = {str(q): {str(h[0]) for h in v} for q, v in pool.items()}
    dup = sum(len(v) - len({str(h[0]) for h in v}) for v in out.values())
    outside = sum(1 for q, v in new.items() for h in v if h[2] is not None and str(h[0]) not in po.get(str(q), set()))
    unsorted = sum(1 for v in out.values() if any(v[i][2] < v[i + 1][2] for i in range(len(v) - 1)))
    rep = {"so_cau": len(out), "trung_trong_cau": dup, "hit_moi_ngoai_pool": outside, "cau_chua_sap": unsorted}
    rep["ok"] = rep["so_cau"] == n_q and dup == 0 and outside == 0 and unsorted == 0
    return rep


def new_rank_dist(out: dict, new: dict, new_ids) -> dict:
    """Hang (1-based) cua hit moi trong q8b_v31, tach doc new_ids va doc cu thieu diem."""
    ni = {str(x) for x in new_ids}
    bands = [(1, 10), (11, 30), (31, 50), (51, 100), (101, 10**9)]
    res = {"new_ids": {}, "cu_thieu": {}}
    for q, v in out.items():
        nset = {str(h[0]) for h in new.get(q, []) if h[2] is not None}
        for r, h in enumerate(v, 1):
            d = str(h[0])
            if d not in nset:
                continue
            g = "new_ids" if d in ni else "cu_thieu"
            for lo, hi in bands:
                if lo <= r <= hi:
                    key = f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}"
                    res[g][key] = res[g].get(key, 0) + 1
    n = max(len(out), 1)
    for g in list(res):
        res[g]["TB_top30_cau"] = round((res[g].get("1-10", 0) + res[g].get("11-30", 0)) / n, 3)
    return res


def rank_band_dist(fhyde: dict, new: dict, top: int = TOP) -> dict:
    """Dem cap moi co diem theo hang fhyde, trong top 1-100, 101-150, 151-200."""
    bands = ((1, 100), (101, 150), (151, 200))
    res = {f"{lo}-{hi}": 0 for lo, hi in bands}
    res["ngoai_top"] = 0
    ranks = {str(q): {str(h[0]): i for i, h in enumerate(v[:top], 1)} for q, v in fhyde.items()}
    for q, hits in new.items():
        for h in hits:
            if len(h) < 3 or h[2] is None:
                continue
            rank = ranks.get(str(q), {}).get(str(h[0]))
            if rank is None:
                res["ngoai_top"] += 1
                continue
            for lo, hi in bands:
                if lo <= rank <= hi:
                    res[f"{lo}-{hi}"] += 1
                    break
    return res


def corr(xs: list, ys: list) -> dict:
    """Pearson, Spearman, |lech| TB/max."""
    import statistics as st
    n = len(xs)
    if n < 2:
        return {"n": n}

    def pear(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else float("nan")

    def rank(a):
        o = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r

    d = [abs(x - y) for x, y in zip(xs, ys)]
    return {"n": n, "pearson": round(pear(xs, ys), 4), "spearman": round(pear(rank(xs), rank(ys)), 4),
            "lech_TB": round(st.mean(d), 4), "lech_max": round(max(d), 4),
            "lech_co_dau_TB": round(st.mean(y - x for x, y in zip(xs, ys)), 4)}


# ---------------------------------------------------------------- b660

def rd(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def wr(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def cmd_select(a):
    import numpy as np
    import pandas as pd
    from bigindex import BigIndex
    from bigpipeline import load_texts
    from boilerplate import BoilerplateFilter
    from export_docs import make_body
    from vi24 import owned_lock

    class PartWriter:
        """Minimal streaming JSONL.zst writer; avoids export_docs optional resource checks on Windows."""
        def __init__(self, out, part_mb=0, log=print):
            import zstandard
            self.out = Path(out)
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.fh = self.out.open("wb")
            self.zw = zstandard.ZstdCompressor(level=6).stream_writer(self.fh)
            self.n = 0
            self.log = log

        def write(self, obj):
            self.zw.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.n += 1

        def close(self):
            self.zw.close()
            self.fh.close()
            self.log(f"ghi {self.out.name}: {self.out.stat().st_size / 1e6:.0f}MB")

    t0 = time.time()
    outd = Path(a.outdir)
    outd.mkdir(parents=True, exist_ok=True)
    with owned_lock(ROOT / "locks/heavy_job.lock", f"q8b3x select pid={os.getpid()}"):
        fhyde = rd(a.fhyde or SELECT_FHYDE)
        pool = rd(a.pool or SELECT_POOL)
        q8b = rd(a.old or SELECT_Q8B)
        new_ids = {str(x) for x in rd(a.new_ids or SELECT_NEW_IDS)}
        pairs = select_pairs(fhyde, pool, q8b, a.top)
        bench = bench_sample(fhyde, q8b, a.bench, top=a.top)
        del pool
        ids = np.load(a.index / "ids.npy")
        pos = {int(d): i for i, d in enumerate(ids)}
        allp = [(q, d) for q, v in pairs.items() for d in v]
        wanted = {d: pos[d] for _, d in allp if d in pos}
        wanted.update({d: pos[d] for _, d, _ in bench if d in pos})
        no_idx = sorted({d for _, d in allp if d not in pos})
        print(f"cap {len(allp)}, doc {len(wanted)}, khong co trong chi muc {len(no_idx)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        idx = BigIndex(a.index)
        bp = BoilerplateFilter.from_json((a.index / "boilerplate.json").read_text(encoding="utf-8"))
        texts = load_texts(idx, wanted, bp, log=lambda m: print(m, flush=True))
        prefix = a.prefix or PREFIX
        w = PartWriter(str(outd / f"{prefix}_docs.jsonl.zst"), 0, log=lambda m: print(m, flush=True))
        empty = 0
        for d, i in sorted(wanted.items()):
            r = texts.get(d)
            if not r:
                continue
            body = make_body(r["title"], r["text"], CHARS, no_title=True)[:CAP]
            empty += not body.strip()
            w.write({"id": d, "idx": i, "text": body})
        w.close()
        no_text = sorted(d for d in wanted if d not in texts)
        q = pd.read_parquet(ROOT / "data/query.parquet")
        with (outd / "queries.jsonl").open("w", encoding="utf-8") as f:
            for i, t in zip(q["id"], q["query"]):
                f.write(json.dumps({"id": int(i), "query": str(t)}, ensure_ascii=False) + "\n")
        mk = lambda dd: [[d, pos[d]] for d in dd if d in pos]
        cands = {q_: mk(v) for q_, v in pairs.items()}
        shards = split_shards(cands, a.shards)
        for k, s in enumerate(shards):
            wr(outd / f"cands_{prefix}_shard{k}.json", s)
        bc = {}
        for q_, d, _ in bench:
            if d in pos:
                bc.setdefault(q_, []).append([d, pos[d]])
        wr(outd / f"cands_{prefix}_bench.json", dict(sorted(bc.items(), key=lambda kv: int(kv[0]))))
        wr(outd / f"{prefix}_bench_old.json", [[q_, d, s] for q_, d, s in bench])
        per_q = [len(v) for v in pairs.values()]
        n_new = sum(str(d) in new_ids for _, d in allp)
        rep = {"top": a.top, "so_cau": len(pairs), "so_cap": len(allp), "so_doc_duy_nhat": len({d for _, d in allp}),
               "cap_cau_TB": round(sum(per_q) / max(len(per_q), 1), 2), "cap_cau_max": max(per_q),
               "cau_0_cap": sum(1 for x in per_q if x == 0),
               "cap_doc_new_ids": n_new, "cap_doc_cu_thieu": len(allp) - n_new,
               "ty_le_new_ids": round(n_new / max(len(allp), 1), 4),
               "doc_ngoai_chi_muc": len(no_idx), "doc_khong_doc_duoc_text": len(no_text),
               "doc_text_rong": empty, "shard_cap": [sum(map(len, s.values())) for s in shards],
               "bench_cap": sum(map(len, bc.values())), "giay": round(time.time() - t0, 1)}
        wr(outd / f"{prefix}_select_report.json", rep)
        print(json.dumps(rep, ensure_ascii=False), flush=True)


def cmd_merge(a):
    old = rd(a.old)
    new = {}
    for p in a.new:
        for q, v in rd(p).items():
            new.setdefault(str(q), []).extend(v)
    pool = rd(a.pool)
    out, st = merge_q8b({str(q): v for q, v in old.items()}, new)
    chk = check_merged(out, new, pool)
    dist = new_rank_dist(out, new, rd(a.new_ids))
    rep = {"merge": st, "kiem": chk, "hang_hit_moi": dist,
           "hit_cu": sum(map(len, old.values())), "hit_moi": sum(1 for v in new.values() for h in v if h[2] is not None),
           "hit_v32": sum(map(len, out.values()))}
    if a.fhyde:
        rep["cap_moi_theo_hang_fhyde"] = rank_band_dist(rd(a.fhyde), new, a.top)
    if a.bench_old and a.bench_new:
        old_rows = rd(a.bench_old)
        new_rows = rd(a.bench_new)
        old_scores = {(str(q), str(d)): score for q, d, score in old_rows}
        got = {(str(q), str(h[0])): h[2] for q, hits in new_rows.items() for h in hits}
        pairs = [(score, got[key]) for key, score in old_scores.items()
                 if key in got and got[key] is not None]
        rep["bench_cung_thang"] = {**corr([x for x, _ in pairs], [y for _, y in pairs]),
                                   "thieu": len(old_scores) - len(pairs)}
    if a.select_report:
        rep["select"] = rd(a.select_report)
    if a.gpu_seconds:
        rep["gpu_gio"] = round(a.gpu_seconds / 3600, 4)
        rep["chi_phi_uoc_USD"] = round(a.gpu_seconds / 3600 * 1.92, 2)
        rep["chi_phi_uoc_voi_10pct_USD"] = round(a.gpu_seconds / 3600 * 1.92 * 1.1, 2)
    rep["cung_thang"] = {"ok": chk["ok"], "so_cau": chk["so_cau"],
                         "trung_trong_cau": chk["trung_trong_cau"],
                         "hit_moi_ngoai_pool": chk["hit_moi_ngoai_pool"],
                         "cau_chua_sap": chk["cau_chua_sap"]}
    assert chk["ok"], chk
    wr(a.out, out)
    if a.report:
        wr(a.report, rep)
    if a.report_extra:
        wr(a.report_extra, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--outdir", required=True)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--bench", type=int, default=200)
    s.add_argument("--index", type=Path, default=IX)
    s.add_argument("--fhyde", default="")
    s.add_argument("--pool", default="")
    s.add_argument("--old", default="")
    s.add_argument("--new-ids", default="")
    s.add_argument("--top", type=int, default=TOP)
    s.add_argument("--prefix", default="")
    m = sp.add_parser("merge")
    m.add_argument("--old", required=True)
    m.add_argument("--new", required=True, nargs="+")
    m.add_argument("--pool", required=True)
    m.add_argument("--new-ids", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--report", default="")
    m.add_argument("--fhyde", default="")
    m.add_argument("--top", type=int, default=TOP)
    m.add_argument("--gpu-seconds", type=float, default=0)
    m.add_argument("--bench-old", default="")
    m.add_argument("--bench-new", default="")
    m.add_argument("--select-report", default="")
    m.add_argument("--report-extra", default="")
    a = ap.parse_args()
    {"select": cmd_select, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
