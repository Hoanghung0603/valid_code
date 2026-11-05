"""Pipeline cho chỉ mục toàn kho (BigIndex): truy vấn -> doc (BM25 băm term) -> chunk nguyên văn -> file nộp.

Hai pha để mỗi shard chỉ phải giải nén một lần:
  1. tìm top doc cho mọi truy vấn trên BigIndex;
  2. gom các doc cần dùng theo shard, đọc shard, làm sạch boilerplate đúng như lúc dựng chỉ mục, chunk,
     rồi BM25 trên chunk của các doc được giữ.

Chỉ truy hồi doc tiếng Việt (chưa dịch truy vấn sang tiếng Trung). Tham số là giá trị khởi đầu, chưa tối ưu
vì chưa có nhãn vàng. Truy vấn được khử trùng term ở cả hai mức để nhất quán với BigIndex.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from bigindex import BigIndex
from bm25 import BM25, tokenize
from boilerplate import BoilerplateFilter
from chunking import chunk_document
from shards import read_shard


def search_docs(idx: BigIndex, queries: list[tuple[int, str]], top_docs: int, rel_ratio: float) -> dict[int, list[tuple[int, int]]]:
    """{qid: [(doc_id, chỉ số trong BigIndex)]} đã lọc theo rel_ratio * điểm cao nhất."""
    lang = idx.meta["lang"]
    out: dict[int, list[tuple[int, int]]] = {}
    for qid, q in queries:
        hits = idx.search(sorted(set(tokenize(q, lang))), k=top_docs)
        if not hits:
            out[int(qid)] = []
            continue
        best = hits[0][1]
        out[int(qid)] = [(idx.doc_id(i), i) for i, s in hits if s >= rel_ratio * best]
    return out


def load_texts(idx: BigIndex, wanted: dict[int, int], bp: BoilerplateFilter, log=print) -> dict[int, dict]:
    """{doc_id: {'host','text'}} với text đã bỏ boilerplate. wanted: {doc_id: chỉ số trong BigIndex}."""
    by_shard: dict[str, set[int]] = defaultdict(set)
    for doc_id, i in wanted.items():
        by_shard[idx.shard_of(i)].add(doc_id)
    texts: dict[int, dict] = {}
    for n, (shard, ids) in enumerate(sorted(by_shard.items()), 1):
        left = set(ids)
        for r in read_shard(Path(shard)):
            did = r.get("id")
            if did in left and r.get("status") == 200:
                left.discard(did)
                host = r.get("host", "")
                title = r.get("title") or ""
                texts[did] = {"host": host, "title": title, "text": bp.clean(host, r.get("text") or "")}
                if not left:
                    break
        if left:
            log(f"CẢNH BÁO: {len(left)} doc không đọc lại được từ {shard}")
        if n % 10 == 0 or n == len(by_shard):
            log(f"đọc shard {n}/{len(by_shard)}, {len(texts)} doc")
    return texts


def pick_chunks(query: str, lang: str, doc_ids: list[int], texts: dict[int, dict],
                chunk_units: int, max_chunks: int) -> list[dict]:
    cand = [(d, c.text) for d in doc_ids if d in texts
            for c in chunk_document(texts[d]["text"], lang, max_units=chunk_units)]
    if not cand:
        return []
    qt = sorted(set(tokenize(query, lang)))
    cidx = BM25().build([tokenize(t, lang) for _, t in cand])
    return [{"doc_id": cand[j][0], "chunk_text": cand[j][1]} for j, _ in cidx.search(qt, k=max_chunks)]


def build_submission(idx: BigIndex, bp: BoilerplateFilter, queries: list[tuple[int, str]], top_docs: int = 10,
                     rel_ratio: float = 0.5, chunk_units: int = 150, max_chunks: int = 5,
                     chunk_docs: int | None = None, log=print) -> list[dict]:
    """chunk_docs: chỉ lấy chunk từ chunk_docs doc đứng đầu (mặc định = tất cả doc được trả về).

    Tách riêng số doc trả về (top_docs, ảnh hưởng Docs F2) và số doc dùng để lấy chunk (ảnh hưởng Chunks F2).
    Chỉ đọc lại text của các doc dùng để lấy chunk.
    """
    lang = idx.meta["lang"]
    t = time.time()
    found = search_docs(idx, queries, top_docs, rel_ratio)
    log(f"pha 1: tìm doc cho {len(queries)} câu ({time.time() - t:.0f}s)")
    cut = chunk_docs if chunk_docs else top_docs
    wanted = {d: i for hits in found.values() for d, i in hits[:cut]}
    log(f"cần đọc {len(wanted)} doc duy nhất để lấy chunk")
    texts = load_texts(idx, wanted, bp, log=log)
    sub = []
    for qid, q in queries:
        doc_ids = [d for d, _ in found[int(qid)]]
        # relevant_docs giữ đủ doc_ids nên mọi doc_id của chunk (lấy từ doc_ids[:cut]) đều nằm trong đó.
        chunks = pick_chunks(q, lang, doc_ids[:cut], texts, chunk_units, max_chunks)
        sub.append({"id": int(qid), "relevant_docs": doc_ids, "relevant_chunks": chunks})
    log(f"pha 2 xong ({time.time() - t:.0f}s)")
    return sub


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-docs", type=int, default=10)
    ap.add_argument("--rel-ratio", type=float, default=0.5)
    ap.add_argument("--chunk-units", type=int, default=150)
    ap.add_argument("--max-chunks", type=int, default=5)
    ap.add_argument("--chunk-docs", type=int, default=None, help="chỉ lấy chunk từ N doc đầu (mặc định: tất cả doc trả về)")
    a = ap.parse_args()

    import pandas as pd
    from evaluate import validate

    idx = BigIndex(Path(a.index))
    bp = BoilerplateFilter.from_json((Path(a.index) / "boilerplate.json").read_text(encoding="utf-8"))
    q = pd.read_parquet(a.queries)
    sub = build_submission(idx, bp, list(zip(q["id"], q["query"])), a.top_docs, a.rel_ratio, a.chunk_units,
                           a.max_chunks, a.chunk_docs, log=lambda m: print(m, flush=True))
    Path(a.out).write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    errs = validate(sub, len(q))
    empty = sum(1 for s in sub if not s["relevant_docs"])
    nd = [len(s["relevant_docs"]) for s in sub]
    nc = [len(s["relevant_chunks"]) for s in sub]
    print(f"ghi {a.out}: {len(sub)} câu, {empty} câu rỗng, doc TB {sum(nd) / len(nd):.2f}, chunk TB {sum(nc) / len(nc):.2f}, "
          f"định dạng {'OK' if not errs else errs[:3]}", flush=True)


if __name__ == "__main__":
    main()
