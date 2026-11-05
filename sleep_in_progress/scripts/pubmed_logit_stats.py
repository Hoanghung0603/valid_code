"""So sánh logit rerank của doc PubMed với logit bge-reranker của doc VI trong cache_pool_rr.json (cùng câu).

Với mỗi câu: logit tại hạng 1/10/30 và trung bình top-1/10/30 của (a) cache PubMed (rerank) và
(b) cache pool VI cũ; in phân bố (mean/p10/p50/p90) + tỉ lệ câu PubMed thắng, ước doc PubMed có
cạnh tranh được với doc VI trong pool hiện tại không.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RANKS = (1, 10, 30)


def quantiles(vals: list) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    def pct(p):
        i = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
        return round(s[i], 4)
    return {"n": len(s), "mean": round(statistics.fmean(s), 4), "p10": pct(10),
            "p50": pct(50), "p90": pct(90), "min": round(s[0], 4), "max": round(s[-1], 4)}


def scores_at(rows: list, k: int):
    """(logit hạng k, TB top-k) của một câu; None nếu không đủ k dòng."""
    vals = [r[2] for r in rows[:k] if r[2] is not None]
    if len(vals) < k:
        return None, (round(sum(vals) / len(vals), 4) if vals else None)
    return round(vals[k - 1], 4), round(sum(vals) / k, 4)


def compare(pm: dict, pool: dict) -> dict:
    out = {"queries_pm": len(pm), "queries_pool": len(pool),
           "queries_both": len(set(pm) & set(pool)), "pm": {}, "pool": {}, "wins": {}}
    common = sorted(set(pm) & set(pool), key=lambda x: int(x))
    for name, cache in (("pm", pm), ("pool", pool)):
        for k in RANKS:
            rk = [scores_at(cache[q], k)[0] for q in common]
            mk = [scores_at(cache[q], k)[1] for q in common]
            out[name][f"rank{k}"] = quantiles([v for v in rk if v is not None])
            out[name][f"mean_top{k}"] = quantiles([v for v in mk if v is not None])
    for k in RANKS:
        both = [(scores_at(pm[q], k)[0], scores_at(pool[q], k)[0]) for q in common]
        both = [(a, b) for a, b in both if a is not None and b is not None]
        out["wins"][f"pm_rank{k}_gt_pool_rank{k}"] = round(
            sum(1 for a, b in both if a > b) / max(len(both), 1), 4)
    # tỉ lệ PubMed hạng k vượt pool hạng 1 (cùng câu)
    for k in RANKS:
        c = 0
        n = 0
        for q in common:
            a = scores_at(pm[q], k)[0]
            b = scores_at(pool[q], 1)[0]
            if a is not None and b is not None:
                n += 1
                c += a > b
        out["wins"][f"pm_rank{k}_gt_pool_top1"] = round(c / max(n, 1), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm", required=True, help="cache rerank PubMed {qid: [[doc_id, idx, logit], ...]}")
    ap.add_argument("--pool", required=True, help="cache_pool_rr.json")
    ap.add_argument("--out", default=None, help="ghi JSON thống kê")
    a = ap.parse_args()
    pm = json.loads(Path(a.pm).read_text(encoding="utf-8"))
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    res = compare(pm, pool)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for src in ("pm", "pool"):
        for k in RANKS:
            for what in (f"rank{k}", f"mean_top{k}"):
                q = res[src][what]
                print(f"{src:4s} {what:12s} n={q.get('n')} mean={q.get('mean')} "
                      f"p10={q.get('p10')} p50={q.get('p50')} p90={q.get('p90')}", flush=True)
    for k, v in res["wins"].items():
        print(f"tỉ lệ {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
