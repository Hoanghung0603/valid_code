"""Vá idx cho cache ZH (từ ids.npy của chỉ mục), so sánh top-30 với #5002, sanity bài nộp zhnew30.

- fix idx: cache {qid: [[doc_id, idx, score], ...]} với idx=-1 (từ z87) -> idx thật trong chỉ mục ZH
  (vị trí của doc_id trong ids.npy), để export_docs.py tra lại shard.
- so sánh: TB số doc trùng trong top-N giữa cache mới và zh_dense_top100.json của #5002.
- sanity bài nộp: đủ 1.200 câu id 1..1200, mọi chunk doc_id ∈ relevant_docs, doc ∈ tập ZH mới (nếu có).

Dạng:
  python zhnew_report.py --rrf rrf.json --ids-npy ids.npy --out-cache rrf_idx.json \
      [--vi vi.json] [--zh zh.json] [--old5002 zh_dense_top100.json] \
      [--sub submission_zhnew30.json] [--allowed-zh new_zh_ids.json] --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def load_id2idx(ids_npy: str) -> dict[int, int]:
    """{doc_id: idx} = vị trí trong ids.npy của chỉ mục (id lệch -> vị trí đầu tiên)."""
    import numpy as np
    ids = np.load(ids_npy)
    out: dict[int, int] = {}
    for i, d in enumerate(ids):
        out.setdefault(int(d), i)
    return out


def fix_idx(cache: dict, id2idx: dict[int, int]) -> tuple[dict, dict]:
    """Ghi idx thật vào hit [doc_id, idx, score]; trả (cache mới, thống kê)."""
    out: dict[str, list] = {}
    n_hits = n_miss = 0
    for q, hits in cache.items():
        rows = []
        for h in hits:
            d = int(h[0])
            idx = id2idx.get(d, -1)
            n_hits += 1
            n_miss += idx < 0
            rows.append([d, idx, h[2]])
        out[str(q)] = rows
    return out, {"n_queries": len(out), "n_hits": n_hits, "n_missing_idx": n_miss}


def top30_overlap(a: dict, b: dict, n: int = 30) -> dict:
    """Số doc trùng trong top-n giữa hai cache {qid: hits} trên các câu chung."""
    ks = [k for k in a if k in b]
    if not ks:
        return {"n_queries": 0}
    ov = [len({int(h[0]) for h in a[k][:n]} & {int(h[0]) for h in b[k][:n]}) for k in ks]
    s = sorted(ov)
    return {
        "n_queries": len(ks),
        "mean": round(sum(ov) / len(ov), 2),
        "median": round(s[len(s) // 2], 2),
        "min": min(ov),
        "max": max(ov),
        "frac_ge_half": round(sum(1 for x in ov if x >= n / 2) / len(ov), 4),
    }


def sub_sanity(sub: list, allowed: set[int] | None = None) -> dict:
    """Kiểm bài nộp: số câu, id 1..1200, chunk doc_id ∈ relevant_docs, doc ∈ allowed."""
    ids = [int(r.get("id", -1)) for r in sub]
    missing_q = [q for q in range(1, 1201) if q not in set(ids)]
    dup_q = len(ids) - len(set(ids))
    n_chunk = 0
    bad_chunk = 0
    n_doc = n_doc_allowed = 0
    docs_per_q = []
    chunks_per_q = []
    for r in sub:
        docs = [int(d) for d in r.get("relevant_docs", [])]
        chunks = r.get("relevant_chunks", [])
        docs_per_q.append(len(docs))
        chunks_per_q.append(len(chunks))
        n_doc += len(docs)
        if allowed is not None:
            n_doc_allowed += sum(1 for d in docs if d in allowed)
        ds = set(docs)
        for c in chunks:
            n_chunk += 1
            if int(c.get("doc_id", -1)) not in ds:
                bad_chunk += 1
    n = max(len(sub), 1)
    out = {
        "n_queries": len(sub),
        "missing_query_ids": missing_q[:20],
        "n_missing_query_ids": len(missing_q),
        "duplicate_query_ids": dup_q,
        "doc_per_q_mean": round(sum(docs_per_q) / n, 2),
        "doc_per_q_min": min(docs_per_q) if docs_per_q else 0,
        "doc_per_q_max": max(docs_per_q) if docs_per_q else 0,
        "chunk_per_q_mean": round(sum(chunks_per_q) / n, 2),
        "chunk_per_q_min": min(chunks_per_q) if chunks_per_q else 0,
        "chunk_per_q_max": max(chunks_per_q) if chunks_per_q else 0,
        "n_chunks": n_chunk,
        "chunk_doc_not_in_relevant_docs": bad_chunk,
    }
    if allowed is not None:
        out["doc_in_new_zh_frac"] = round(n_doc_allowed / n_doc, 6) if n_doc else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf", default="", help="cache RRF (idx=-1) để vá idx")
    ap.add_argument("--ids-npy", default="", help="ids.npy của chỉ mục ZH")
    ap.add_argument("--out-cache", default="", help="cache đã vá idx")
    ap.add_argument("--vi", default="")
    ap.add_argument("--zh", default="")
    ap.add_argument("--old5002", default="", help="zh_dense_top100.json của #5002")
    ap.add_argument("--sub", default="")
    ap.add_argument("--allowed-zh", default="")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rep: dict = {}
    if a.rrf and a.ids_npy:
        if not a.out_cache:
            raise SystemExit("cần --out-cache khi --rrf")
        cache = json.loads(Path(a.rrf).read_text(encoding="utf-8"))
        fixed, st = fix_idx(cache, load_id2idx(a.ids_npy))
        Path(a.out_cache).write_text(json.dumps(fixed), encoding="utf-8")
        rep["fix_idx"] = st
        print(f"vá idx: {st}", flush=True)

    if a.old5002:
        old = json.loads(Path(a.old5002).read_text(encoding="utf-8"))
        rep["overlap_top30"] = {}
        srcs = {}
        if a.rrf:
            srcs["rrf_vs_5002"] = json.loads(Path(a.rrf).read_text(encoding="utf-8"))
        if a.vi:
            srcs["vi_vs_5002"] = json.loads(Path(a.vi).read_text(encoding="utf-8"))
        if a.zh:
            srcs["zh_vs_5002"] = json.loads(Path(a.zh).read_text(encoding="utf-8"))
        for name, cache in srcs.items():
            rep["overlap_top30"][name] = top30_overlap(cache, old, a.n)
            print(f"{name}: {rep['overlap_top30'][name]}", flush=True)

    if a.sub:
        allowed = None
        if a.allowed_zh:
            obj = json.loads(Path(a.allowed_zh).read_text(encoding="utf-8"))
            allowed = {int(x) for x in (obj if isinstance(obj, list) else obj.get("ids", []))}
        sub = json.loads(Path(a.sub).read_text(encoding="utf-8"))
        rep["sub_sanity"] = sub_sanity(sub, allowed)
        print(f"sub_sanity: {rep['sub_sanity']}", flush=True)

    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ghi {a.out}", flush=True)


if __name__ == "__main__":
    main()
