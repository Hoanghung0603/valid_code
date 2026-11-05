"""Thống kê phép thử dense đa ngữ VI -> ZH (bge-m3): phân bố cosine, host, ví dụ.

Đầu vào: --zh cache {qid: [[doc_id, idx, cosine], ...]} giảm dần (zh_dense_top100.json của embed_all.py),
tùy chọn --vi cache cùng định dạng (dense_full_top200.json) để so sánh, --queries JSONL {id, query},
--index chỉ mục ZH (BigIndex) để lấy host/tiêu đề của top-K doc ZH (đọc lại shard, như bigpipeline.load_texts).

In: phân vị cosine top-1 và top-K (hạng K) cho ZH và VI; tỉ lệ câu có cosine top-1 ZH > top-1 VI;
top host ZH trong top-K; --examples câu ví dụ (truy vấn + tiêu đề --ex-docs doc ZH đầu).
Ghi --out JSON nếu có.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def percentiles(xs: list, ps=(0, 10, 25, 50, 75, 90, 100)) -> dict:
    """Phân vị kiểu nearest-rank trên danh sách số (rỗng -> {})."""
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    out = {}
    for p in ps:
        k = min(n - 1, max(0, round(p / 100 * (n - 1))))
        out[f"p{p}"] = round(float(s[k]), 4)
    out["mean"] = round(sum(s) / n, 4)
    return out


def rank_scores(cache: dict, rank: int) -> dict:
    """{qid: cosine ở hạng rank (1-based)}; câu thiếu hạng đó bị bỏ."""
    return {str(q): float(h[rank - 1][2]) for q, h in cache.items() if len(h) >= rank and h[rank - 1][2] is not None}


def score_dist(cache: dict, k: int) -> dict:
    return {"top1": percentiles(list(rank_scores(cache, 1).values())),
            f"top{k}": percentiles(list(rank_scores(cache, k).values()))}


def compare_top1(zh: dict, vi: dict) -> dict:
    """So cosine top-1 ZH với top-1 VI trên các câu chung."""
    a, b = rank_scores(zh, 1), rank_scores(vi, 1)
    common = [q for q in a if q in b]
    if not common:
        return {"n": 0}
    diff = [a[q] - b[q] for q in common]
    return {"n": len(common), "zh_gt_vi": sum(1 for d in diff if d > 0),
            "diff": percentiles(diff, (10, 50, 90))}


def top_docs(cache: dict, k: int) -> dict:
    """{doc_id: idx} của mọi doc trong top-k các câu."""
    return {int(h[0]): int(h[1]) for hits in cache.values() for h in hits[:k]}


def host_counts(cache: dict, hosts: dict, k: int, n: int = 15) -> list:
    """[(host, số lần xuất hiện trong top-k, tỉ lệ)] n host nhiều nhất."""
    c = Counter(hosts.get(int(h[0]), "?") for hits in cache.values() for h in hits[:k])
    tot = sum(c.values()) or 1
    return [(hst, m, round(m / tot, 4)) for hst, m in c.most_common(n)]


def examples(cache: dict, queries: dict, titles: dict, qids: list, ndocs: int = 3) -> list:
    out = []
    for q in qids:
        hits = cache.get(str(q), [])[:ndocs]
        out.append({"id": str(q), "query": queries.get(str(q), ""),
                    "docs": [[int(h[0]), round(float(h[2]), 4), titles.get(int(h[0]), "")] for h in hits]})
    return out


def pick_example_qids(cache: dict, n: int) -> list:
    """n câu trải đều theo cosine top-1 (cao, giữa, thấp) để kiểm bằng mắt."""
    s = sorted(rank_scores(cache, 1).items(), key=lambda kv: -kv[1])
    if not s or n <= 0:
        return []
    if n == 1:
        return [s[0][0]]
    return [s[round(i * (len(s) - 1) / (n - 1))][0] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", required=True)
    ap.add_argument("--vi", default="")
    ap.add_argument("--queries", default="")
    ap.add_argument("--index", default="", help="chỉ mục ZH để lấy host/tiêu đề top-K")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--ex-docs", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    zh = json.loads(Path(a.zh).read_text(encoding="utf-8"))
    res = {"n_queries": len(zh), "zh": score_dist(zh, a.k)}
    print(f"ZH: {len(zh)} câu; cosine {res['zh']}", flush=True)
    if a.vi:
        vi = json.loads(Path(a.vi).read_text(encoding="utf-8"))
        res["vi"] = score_dist(vi, a.k)
        res["zh_vs_vi_top1"] = compare_top1(zh, vi)
        print(f"VI: cosine {res['vi']}", flush=True)
        print(f"top-1 ZH vs VI: {res['zh_vs_vi_top1']}", flush=True)

    queries = {}
    if a.queries:
        for ln in Path(a.queries).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                queries[str(r["id"])] = r["query"]

    hosts, titles = {}, {}
    if a.index:
        from bigindex import BigIndex
        from bigpipeline import load_texts
        from boilerplate import BoilerplateFilter
        idx = BigIndex(Path(a.index))
        bp = BoilerplateFilter.from_json((Path(a.index) / "boilerplate.json").read_text(encoding="utf-8"))
        texts = load_texts(idx, top_docs(zh, a.k), bp, log=lambda m: print(m, flush=True))
        hosts = {d: t["host"] for d, t in texts.items()}
        titles = {d: (t["title"] or t["text"][:80]).replace("\n", " ")[:120] for d, t in texts.items()}
        res["n_docs_topk"] = len(top_docs(zh, a.k))
        res["hosts_topk"] = host_counts(zh, hosts, a.k)
        print(f"{res['n_docs_topk']} doc ZH duy nhất trong top-{a.k}; top host:", flush=True)
        for h, m, r in res["hosts_topk"]:
            print(f"  {h}\t{m}\t{r}", flush=True)

    res["examples"] = examples(zh, queries, titles, pick_example_qids(zh, a.examples), a.ex_docs)
    for e in res["examples"]:
        print(f"[{e['id']}] {e['query']}", flush=True)
        for d, s, t in e["docs"]:
            print(f"    {d}\t{s}\t{t}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"ghi {a.out}", flush=True)


if __name__ == "__main__":
    main()
