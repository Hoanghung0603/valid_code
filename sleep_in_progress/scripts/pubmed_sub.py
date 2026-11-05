"""Dựng bài thăm dò PubMed: mỗi câu lấy top-30 doc theo cache, chunk = title + "\\n" + abstract nguyên văn.

relevant_docs = 30 doc_id đầu theo thứ hạng cache; relevant_chunks = đúng 30 chunk của 30 doc đó
(mọi chunk doc_id ∈ relevant_docs). Dùng cho submission_pm_dense30 (cache RRF) và pm_rr30 (cache rerank).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pubmed_docs import doc_text, is_empty_doc, iter_texts


def load_texts(paths: list) -> dict:
    """{id: (title, abstract)} từ pubmed_text.jsonl(.zst)."""
    out = {}
    for r in iter_texts([Path(p) for p in paths]):
        out[int(r["id"])] = (r.get("title") or "", r.get("abstract") or "")
    return out


def build_sub(cache: dict, texts: dict, top: int = 30) -> list:
    """cache {qid: [[doc_id, idx, score], ...]} (giảm dần) -> list bài nộp theo id tăng dần.

    Lấy top doc_id ĐẦU TIÊN DUY NHẤT (bỏ dòng trùng doc_id); chunk = title + "\\n" + abstract nguyên văn.
    """
    items = []
    for qid in sorted(cache, key=lambda x: int(x)):
        docs, chunks = [], []
        for row in cache[qid]:
            if len(docs) >= top:
                break
            d = int(row[0])
            if d in docs:
                continue
            title, abstract = texts.get(d, ("", ""))
            docs.append(d)
            chunks.append({"doc_id": d, "chunk_text": doc_text(title, abstract),
                           "chunk_order": len(chunks)})
        items.append({"id": int(qid), "relevant_docs": docs, "relevant_chunks": chunks})
    return items


def check_sub(items: list, top: int, n_queries: int) -> list:
    """Trả danh sách lỗi (rỗng = hợp lệ)."""
    errs = []
    if len(items) != n_queries:
        errs.append(f"{len(items)} câu, cần {n_queries}")
    for it in items:
        qid = it["id"]
        docs = it["relevant_docs"]
        if len(docs) > top:
            errs.append(f"id={qid}: {len(docs)} doc > {top}")
        if len(set(docs)) != len(docs):
            errs.append(f"id={qid}: doc trùng")
        dset = set(docs)
        for ch in it["relevant_chunks"]:
            if ch["doc_id"] not in dset:
                errs.append(f"id={qid}: chunk doc_id {ch['doc_id']} không trong relevant_docs")
        if len(it["relevant_chunks"]) != len(docs):
            errs.append(f"id={qid}: {len(it['relevant_chunks'])} chunk != {len(docs)} doc")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="JSON {qid: [[doc_id, idx, score], ...]} giảm dần")
    ap.add_argument("--texts", required=True, nargs="+", help="pubmed_text.jsonl.zst")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--n-queries", type=int, default=1200)
    a = ap.parse_args()
    t0 = time.time()
    cache = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    texts = load_texts(a.texts)
    print(f"cache {len(cache)} câu, {len(texts)} bài PubMed ({time.time() - t0:.0f}s)", flush=True)
    items = build_sub(cache, texts, a.top)
    errs = check_sub(items, a.top, a.n_queries)
    if errs:
        print("LỖI:\n" + "\n".join(errs[:30]), flush=True)
        raise SystemExit(1)
    n_doc = sum(len(it["relevant_docs"]) for it in items) / max(len(items), 1)
    n_ab = sum(1 for it in items for ch in it["relevant_chunks"]
               if (texts.get(ch["doc_id"], ("", ""))[1] or "").strip())
    n_ch = sum(len(it["relevant_chunks"]) for it in items)
    Path(a.out).write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"ghi {a.out}: {len(items)} câu, TB {n_doc:.1f} doc/câu, "
          f"{n_ch} chunk ({n_ab} có abstract), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
