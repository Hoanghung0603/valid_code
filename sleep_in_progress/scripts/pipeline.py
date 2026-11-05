"""Pipeline tối thiểu: truy vấn -> BM25 mức tài liệu -> chọn chunk nguyên văn -> file nộp.

Giai đoạn này CHỈ truy hồi tài liệu tiếng Việt (truy vấn là tiếng Việt, chưa có bước dịch sang tiếng Trung),
nên tài liệu tiếng Trung chưa được trả về. Tham số (top_docs, rel_ratio, chunk size...) chỉ là giá trị khởi đầu,
chưa được tối ưu vì chưa có nhãn vàng.
"""
import argparse
import json
import sys
from pathlib import Path

from bm25 import BM25, tokenize
from boilerplate import BoilerplateFilter
from chunking import chunk_document
from corpus import iter_docs
from evaluate import validate


class Retriever:
    def __init__(self, docs: list[dict], lang: str = "vi", chunk_units: int = 150):
        self.lang, self.chunk_units = lang, chunk_units
        docs = [d for d in docs if d["lang"] == lang]
        bp = BoilerplateFilter().fit(docs)
        self.docs = [{**d, "text": bp.clean(d["host"], d["text"])} for d in docs]
        self.index = BM25().build([tokenize(d["title"] + "\n" + d["text"], lang) for d in self.docs])

    def retrieve(self, query: str, top_docs: int = 10, rel_ratio: float = 0.5,
                 max_chunks: int = 5) -> dict:
        """Trả về {'relevant_docs': [id], 'relevant_chunks': [{'doc_id','chunk_text'}]}.

        Giữ tài liệu có điểm >= rel_ratio * điểm cao nhất (F2 ưu tiên Recall nhưng vẫn cần Precision).
        Chunk: BM25 trên các chunk của tài liệu được giữ, lấy max_chunks tốt nhất.
        """
        qt = tokenize(query, self.lang)
        hits = self.index.search(qt, k=top_docs)
        if not hits:
            return {"relevant_docs": [], "relevant_chunks": []}
        best = hits[0][1]
        kept = [(i, s) for i, s in hits if s >= rel_ratio * best]
        doc_ids = [self.docs[i]["id"] for i, _ in kept]

        cand = [(self.docs[i]["id"], c.text) for i, _ in kept
                for c in chunk_document(self.docs[i]["text"], self.lang, max_units=self.chunk_units)]
        chunks = []
        if cand:
            cidx = BM25().build([tokenize(t, self.lang) for _, t in cand])
            chunks = [{"doc_id": cand[j][0], "chunk_text": cand[j][1]} for j, _ in cidx.search(qt, k=max_chunks)]
        return {"relevant_docs": doc_ids, "relevant_chunks": chunks}


def build_submission(retriever: Retriever, queries: list[tuple[int, str]], **kw) -> list[dict]:
    return [{"id": int(qid), **retriever.retrieve(q, **kw)} for qid, q in queries]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="thư mục chứa *.jsonl.zst")
    ap.add_argument("--queries", default="data/query.parquet")
    ap.add_argument("--out", default="data/submission.json")
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--top-docs", type=int, default=10)
    ap.add_argument("--rel-ratio", type=float, default=0.5)
    ap.add_argument("--max-chunks", type=int, default=5)
    a = ap.parse_args()

    import pandas as pd
    q = pd.read_parquet(a.queries)
    docs = list(iter_docs(Path(a.shards), limit=a.limit_docs))
    print(f"{len(docs)} tài liệu đọc được", file=sys.stderr)
    r = Retriever(docs)
    sub = build_submission(r, list(zip(q["id"], q["query"])), top_docs=a.top_docs,
                           rel_ratio=a.rel_ratio, max_chunks=a.max_chunks)
    Path(a.out).write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    errs = validate(sub, len(q))
    empty = sum(1 for s in sub if not s["relevant_docs"])
    print(f"ghi {a.out}: {len(sub)} câu, {empty} câu rỗng, định dạng {'OK' if not errs else errs[:3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
