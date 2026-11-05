"""So sánh kho URL cũ (links_corpus.parquet) với kho mới của BTC (links_corpus_vi_en_zh.parquet, 12/10)
và đo tỉ lệ doc trong một bài nộp còn nằm trong kho mới.

Dùng:
  python corpus_diff.py --old OLD.parquet --new NEW.parquet --sub SUB.json [--sub SUB2.json] --out REPORT.json
"""
from __future__ import annotations

import argparse
import collections
import json
import urllib.parse

import pyarrow.parquet as pq


def load_corpus(path: str) -> dict[int, str]:
    d = pq.read_table(path, columns=["id", "url"]).to_pydict()
    return dict(zip(d["id"], d["url"]))


def host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def diff_corpus(old: dict[int, str], new: dict[int, str]) -> dict:
    """Thống kê id giữ / bỏ / thêm, id giữ nhưng đổi URL, theo host."""
    old_ids, new_ids = set(old), set(new)
    kept = old_ids & new_ids
    changed = sum(1 for i in kept if old[i] != new[i])
    dropped = old_ids - new_ids
    added = new_ids - old_ids
    # URL trùng nhưng id khác (đánh số lại)
    old_by_url = {u: i for i, u in old.items()}
    added_url_in_old = sum(1 for i in added if new[i] in old_by_url)
    h_old = collections.Counter(host(old[i]) for i in old_ids)
    h_new = collections.Counter(host(new[i]) for i in new_ids)
    h_kept = collections.Counter(host(old[i]) for i in kept)
    hosts = sorted(set(h_old) | set(h_new), key=lambda k: -max(h_old[k], h_new[k]))
    by_host = [
        {"host": k, "old": h_old[k], "new": h_new[k], "kept": h_kept[k]}
        for k in hosts
    ]
    return {
        "old_n": len(old_ids),
        "new_n": len(new_ids),
        "kept": len(kept),
        "kept_url_changed": changed,
        "dropped": len(dropped),
        "added": len(added),
        "added_url_was_in_old": added_url_in_old,
        "by_host": by_host,
    }


def sub_membership(sub: list[dict], new_ids: set[int], old: dict[int, str]) -> dict:
    """Tỉ lệ doc / chunk của bài nộp nằm trong kho mới, theo hạng doc."""
    n_doc = n_in = 0
    n_chunk = n_chunk_in = 0
    per_q_in = []
    band = collections.defaultdict(lambda: [0, 0])  # hạng -> [tổng, trong kho mới]
    out_hosts = collections.Counter()
    for r in sub:
        docs = r["relevant_docs"]
        k = 0
        for rank, d in enumerate(docs, 1):
            b = (rank - 1) // 30
            band[b][0] += 1
            n_doc += 1
            if d in new_ids:
                n_in += 1
                k += 1
                band[b][1] += 1
            else:
                out_hosts[host(old.get(d, ""))] += 1
        per_q_in.append(k / len(docs) if docs else 0.0)
        for c in r["relevant_chunks"]:
            n_chunk += 1
            n_chunk_in += c["doc_id"] in new_ids
    return {
        "doc_avg_per_q": n_doc / len(sub),
        "doc_in_new_frac": n_in / n_doc if n_doc else 0.0,
        "doc_in_new_avg_per_q": n_in / len(sub),
        "macro_in_frac": sum(per_q_in) / len(per_q_in),
        "chunk_in_new_frac": n_chunk_in / n_chunk if n_chunk else 0.0,
        "by_rank30": {f"{b*30+1}-{b*30+30}": [t, i, round(i / t, 4)] for b, (t, i) in sorted(band.items())},
        "out_hosts_top": out_hosts.most_common(20),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--sub", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old = load_corpus(a.old)
    new = load_corpus(a.new)
    rep = {"corpus": diff_corpus(old, new), "subs": {}}
    new_ids = set(new)
    for p in a.sub:
        with open(p, encoding="utf-8") as f:
            sub = json.load(f)
        rep["subs"][p] = sub_membership(sub, new_ids, old)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    c = rep["corpus"]
    print({k: v for k, v in c.items() if k != "by_host"})
    for p, s in rep["subs"].items():
        print(p, {k: v for k, v in s.items() if k not in ("by_rank30", "out_hosts_top")})


if __name__ == "__main__":
    main()
